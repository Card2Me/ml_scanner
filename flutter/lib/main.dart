import 'dart:async';

import 'dart:ui' as ui;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;

import 'services/segmentation_service.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MlScannerApp());
}

class MlScannerApp extends StatelessWidget {
  const MlScannerApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ML Scanner',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal),
        useMaterial3: true,
      ),
      home: const ScannerScreen(),
    );
  }
}

class ScannerScreen extends StatefulWidget {
  const ScannerScreen({super.key});

  @override
  State<ScannerScreen> createState() => _ScannerScreenState();
}

class _ScannerScreenState extends State<ScannerScreen>
    with SingleTickerProviderStateMixin {
  static const int _cornerSampleCount = 32;
  final SegmentationService _segmentationService = SegmentationService();
  CameraController? _cameraController;
  Future<void>? _initialization;
  SegmentationResult? _latestResult;
  CameraImage? _latestCameraImage;
  bool _isProcessing = false;
  String? _error;
  bool _isStreaming = false;
  DateTime _lastSegmentation = DateTime.fromMillisecondsSinceEpoch(0);
  final Duration _segmentationInterval = const Duration(milliseconds: 350);
  List<Offset>? _outlineSegments;
  late final AnimationController _cornerController;
  List<Offset>? _startCorners;
  List<Offset>? _targetCorners;

  @override
  void initState() {
    super.initState();
    _cornerController =
        AnimationController(
          vsync: this,
          duration: const Duration(milliseconds: 350),
        )..addListener(() {
          setState(() {});
        });
    _initialization = _initialize();
  }

  @override
  void dispose() {
    unawaited(_stopImageStream());
    _cameraController?.dispose();
    _segmentationService.dispose();
    _cornerController.dispose();
    super.dispose();
  }

  Future<void> _initialize() async {
    final cameras = await availableCameras();
    if (cameras.isEmpty) {
      throw StateError('No camera available on this device');
    }

    final controller = CameraController(
      cameras.first,
      ResolutionPreset.low,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );
    _cameraController = controller;
    await controller.initialize();
    await _startImageStream();
    await _segmentationService.initialize();
  }

  Future<void> _startImageStream() async {
    final controller = _cameraController;
    if (controller == null ||
        _isStreaming ||
        controller.value.isStreamingImages) {
      return;
    }
    await controller.startImageStream((image) {
      _latestCameraImage = image;
      if (_isProcessing) {
        return;
      }
      unawaited(_processCameraImage(image));
    });
    _isStreaming = true;
  }

  Future<void> _stopImageStream() async {
    final controller = _cameraController;
    if (controller == null || !_isStreaming) {
      return;
    }
    await controller.stopImageStream();
    _isStreaming = false;
  }

  Future<void> _processCameraImage(CameraImage image) async {
    if (_isProcessing) {
      return;
    }
    _isProcessing = true;
    try {
      final converted = _convertYuv420ToImage(image);
      final adjusted = _applyOrientation(converted);
      final result = await _segmentationService.segmentImage(adjusted);
      if (!mounted) {
        return;
      }
      _updateResult(result);
    } catch (error) {
      if (!mounted) {
        return;
      }
      setState(() {
        _error = error.toString();
      });
    } finally {
      _isProcessing = false;
    }
  }

  void _updateResult(SegmentationResult result) {
    setState(() {
      _latestResult = result;
      _error = null;
      _outlineSegments = _buildOutlineSegments(result);
    });
    _updateCorners(result);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: FutureBuilder<void>(
        future: _initialization,
        builder: (context, snapshot) {
          if (snapshot.connectionState != ConnectionState.done) {
            return const Center(child: CircularProgressIndicator());
          }
          if (snapshot.hasError) {
            return _ErrorState(message: snapshot.error.toString());
          }
          return SafeArea(
            child: Column(
              children: [
                _buildCameraPreview(),
                Expanded(child: _buildResultPanel()),
              ],
            ),
          );
        },
      ),
    );
  }

  Widget _buildCameraPreview() {
    final controller = _cameraController;
    if (controller == null || !controller.value.isInitialized) {
      return Container(
        width: double.infinity,
        color: Colors.black,
        padding: const EdgeInsets.all(24),
        child: const Center(
          child: Text(
            'Camera is not ready.',
            style: TextStyle(color: Colors.white70),
          ),
        ),
      );
    }

    final previewSize = controller.value.previewSize;
    final aspectRatio = previewSize != null
        ? (previewSize.height / previewSize.width)
        : controller.value.aspectRatio;

    return Container(
      width: double.infinity,
      color: Colors.black,
      child: AspectRatio(
        aspectRatio: aspectRatio,
        child: Stack(
          fit: StackFit.expand,
          children: [
            CameraPreview(controller),
            if (_latestResult != null)
              IgnorePointer(
                child: FittedBox(
                  fit: BoxFit.cover,
                  child: SizedBox(
                    width: _latestResult!.maskWidth.toDouble(),
                    height: _latestResult!.maskHeight.toDouble(),
                    child: Stack(
                      fit: StackFit.expand,
                      children: [
                        Opacity(
                          opacity: 0.3,
                          child: Image.memory(
                            _latestResult!.maskBytes,
                            fit: BoxFit.fill,
                            gaplessPlayback: true,
                            filterQuality: FilterQuality.low,
                          ),
                        ),
                        if (_outlineSegments != null)
                          CustomPaint(
                            painter: _OutlinePainter(
                              segments: _outlineSegments!,
                              color: Colors.redAccent,
                            ),
                          ),
                      ],
                    ),
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildResultPanel() {
    if (_latestResult == null) {
      return Container(
        width: double.infinity,
        color: Colors.grey.shade100,
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              'Live segmentation',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 4),
            const Text('카메라 영상에서 자동으로 문서 윤곽을 추적합니다.'),
            if (_error != null) ...[
              const SizedBox(height: 8),
              Text(_error!, style: const TextStyle(color: Colors.red)),
            ],
            const SizedBox(height: 12),
            Align(
              alignment: Alignment.centerRight,
              child: OutlinedButton.icon(
                onPressed: _clearResult,
                icon: const Icon(Icons.clear),
                label: const Text('Clear'),
              ),
            ),
          ],
        ),
      );
    }

    final result = _latestResult!;
    return Container(
      width: double.infinity,
      color: Colors.grey.shade50,
      padding: const EdgeInsets.all(16),
      child: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  'Latest segmentation',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                Text(_formatDuration(result.totalTime)),
              ],
            ),
            const SizedBox(height: 8),
            Text('Mask: ${result.maskWidth}x${result.maskHeight}'),
            const SizedBox(height: 8),
            _MetricsWrap(result: result, formatter: _formatDuration),

            if (_error != null) ...[
              const SizedBox(height: 8),
              Text(_error!, style: const TextStyle(color: Colors.red)),
            ],
            const SizedBox(height: 12),
            Align(
              alignment: Alignment.centerRight,
              child: OutlinedButton.icon(
                onPressed: _clearResult,
                icon: const Icon(Icons.clear),
                label: const Text('Clear'),
              ),
            ),
          ],
        ),
      ),
    );
  }

  void _clearResult() {
    setState(() {
      _latestResult = null;
      _error = null;
    });
    _startCorners = null;
    _targetCorners = null;
    _cornerController.stop();
  }

  /// Updates the target contour and kicks off the morphing animation whenever
  /// a confident set of corners arrives from the segmentation service.
  void _updateCorners(SegmentationResult result) {
    if (result.corners == null ||
        result.maskWidth == 0 ||
        result.maskHeight == 0 ||
        result.confidence < 0.74) {
      _startCorners = null;
      _targetCorners = null;
      _cornerController.stop();
      return;
    }
    final normalized = result.corners!
        .map(
          (corner) =>
              Offset(corner.x / result.maskWidth, corner.y / result.maskHeight),
        )
        .toList();
    final current = _targetCorners == null
        ? normalized
        : (_animatedCorners ?? _targetCorners!);
    _startCorners = current;
    _targetCorners = normalized;
    _cornerController.forward(from: 0);
  }

  List<Offset>? get _animatedCorners {
    final target = _targetCorners;
    if (target == null) {
      return null;
    }
    if (target.length < 3) {
      return null;
    }
    final start = _startCorners ?? target;
    final t = _cornerController.value;
    final count = target.length;
    final List<Offset> interpolated = [];
    for (var i = 0; i < count; i++) {
      final startCorner = start[i % start.length];
      final targetCorner = target[i];
      final dx =
          ui.lerpDouble(startCorner.dx, targetCorner.dx, t) ?? targetCorner.dx;
      final dy =
          ui.lerpDouble(startCorner.dy, targetCorner.dy, t) ?? targetCorner.dy;
      interpolated.add(Offset(dx, dy));
    }
    return interpolated;
  }

  List<Offset>? _resampleCorners(
    List<Corner> corners,
    int width,
    int height, {
    int sampleCount = 32,
  }) {
    if (corners.length < 3 || width == 0 || height == 0) {
      return null;
    }
    final normalized = corners
        .map((corner) => Offset(corner.x / width, corner.y / height))
        .toList();
    final closed = [...normalized, normalized.first];
    final lengths = <double>[];
    double total = 0;
    for (var i = 0; i < closed.length - 1; i++) {
      final segmentLength = (closed[i + 1] - closed[i]).distance;
      lengths.add(segmentLength);
      total += segmentLength;
    }
    if (total == 0) {
      return null;
    }
    final step = total / sampleCount;
    final result = <Offset>[];
    double targetDist = 0;
    int segmentIndex = 0;
    double accumulated = lengths.isNotEmpty ? lengths[0] : 0;
    for (var i = 0; i < sampleCount; i++) {
      while (targetDist > accumulated && segmentIndex < lengths.length - 1) {
        segmentIndex++;
        accumulated += lengths[segmentIndex];
      }
      final segmentStart = closed[segmentIndex];
      final segmentEnd = closed[segmentIndex + 1];
      final prevAccumulated = accumulated - lengths[segmentIndex];
      final localT = lengths[segmentIndex] == 0
          ? 0
          : (targetDist - prevAccumulated) / lengths[segmentIndex];
      final point = Offset(
        ui.lerpDouble(segmentStart.dx, segmentEnd.dx, localT.toDouble()) ??
            segmentStart.dx,
        ui.lerpDouble(segmentStart.dy, segmentEnd.dy, localT.toDouble()) ??
            segmentStart.dy,
      );
      result.add(point);
      targetDist += step;
    }
    return result;
  }

  List<Offset>? _buildOutlineSegments(SegmentationResult result) {
    final corners = result.corners;
    if (corners == null || corners.length < 3) {
      return null;
    }
    final normals = corners
        .map(
          (corner) =>
              Offset(corner.x / result.maskWidth, corner.y / result.maskHeight),
        )
        .toList();
    return normals;
  }

  String _formatDuration(Duration duration) {
    return '${duration.inMilliseconds} ms';
  }

  img.Image _applyOrientation(img.Image image) {
    final controller = _cameraController;
    if (controller == null) {
      return image;
    }
    final rotation = controller.description.sensorOrientation;
    switch (rotation) {
      case 90:
        return img.copyRotate(image, angle: 90);
      case 180:
        return img.copyRotate(image, angle: 180);
      case 270:
        return img.copyRotate(image, angle: 270);
      default:
        return image;
    }
  }
}

class _ErrorState extends StatelessWidget {
  const _ErrorState({required this.message});

  final String message;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Text(
          message,
          style: Theme.of(
            context,
          ).textTheme.bodyLarge?.copyWith(color: Colors.red),
          textAlign: TextAlign.center,
        ),
      ),
    );
  }
}

class _MetricsWrap extends StatelessWidget {
  const _MetricsWrap({required this.result, required this.formatter});

  final SegmentationResult result;
  final String Function(Duration) formatter;

  @override
  Widget build(BuildContext context) {
    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: [
        _InfoChip(label: '전체', value: formatter(result.totalTime)),
        _InfoChip(label: '전처리', value: formatter(result.preprocessTime)),
        _InfoChip(label: '추론', value: formatter(result.inferenceTime)),
        _InfoChip(label: '후처리', value: formatter(result.postprocessTime)),
        _InfoChip(
          label: '확신도',
          value: '${(result.confidence * 100).toStringAsFixed(1)}%',
        ),
        _InfoChip(label: '가속', value: result.accelerated ? 'ON' : 'OFF'),
      ],
    );
  }
}

class _InfoChip extends StatelessWidget {
  const _InfoChip({required this.label, required this.value});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Chip(
      label: Text('$label: $value'),
      side: BorderSide(color: Colors.teal.shade200),
      backgroundColor: Colors.teal.shade50,
    );
  }
}

class _OutlinePainter extends CustomPainter {
  const _OutlinePainter({required this.segments, this.color});

  final List<Offset> segments;
  final Color? color;

  @override
  void paint(Canvas canvas, Size size) {
    if (segments.length < 3) {
      return;
    }
    final paint = Paint()
      ..color = color ?? Colors.redAccent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3;

    final path = Path();
    for (var i = 0; i < segments.length; i++) {
      final point = segments[i];
      final dx = point.dx * size.width;
      final dy = point.dy * size.height;
      if (i == 0) {
        path.moveTo(dx, dy);
      } else {
        path.lineTo(dx, dy);
      }
    }
    path.close();
    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant _OutlinePainter oldDelegate) {
    return oldDelegate.segments != segments || oldDelegate.color != color;
  }
}

img.Image _convertYuv420ToImage(CameraImage image) {
  final width = image.width;
  final height = image.height;
  final img.Image output = img.Image(width: width, height: height);

  final yPlane = image.planes[0];
  final uPlane = image.planes[1];
  final vPlane = image.planes[2];
  final yBytes = yPlane.bytes;
  final uBytes = uPlane.bytes;
  final vBytes = vPlane.bytes;
  final yRowStride = yPlane.bytesPerRow;
  final uRowStride = uPlane.bytesPerRow;
  final vRowStride = vPlane.bytesPerRow;
  final uPixelStride = uPlane.bytesPerPixel ?? 1;
  final vPixelStride = vPlane.bytesPerPixel ?? 1;

  for (var y = 0; y < height; y++) {
    for (var x = 0; x < width; x++) {
      final yIndex = y * yRowStride + x;
      final uvRow = y ~/ 2;
      final uvCol = x ~/ 2;
      final uIndex = uvRow * uRowStride + uvCol * uPixelStride;
      final vIndex = uvRow * vRowStride + uvCol * vPixelStride;

      final yValue = yBytes[yIndex];
      final uValue = uBytes[uIndex];
      final vValue = vBytes[vIndex];

      final r = (yValue + 1.402 * (vValue - 128)).round().clamp(0, 255);
      final g = (yValue - 0.344136 * (uValue - 128) - 0.714136 * (vValue - 128))
          .round()
          .clamp(0, 255);
      final b = (yValue + 1.772 * (uValue - 128)).round().clamp(0, 255);
      output.setPixelRgba(x, y, r, g, b, 255);
    }
  }

  return output;
}
