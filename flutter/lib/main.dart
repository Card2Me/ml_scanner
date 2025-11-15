import 'dart:async';
import 'dart:io';

import 'dart:ui' as ui;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';

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

  // New state variables
  bool _isFlashOn = false;
  bool _isTwoPageMode = false;
  String _selectedFolder = '기본';
  final List<String> _folders = ['기본'];

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
    _loadFolders();
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

  String? _getValidationMessage(SegmentationResult? result) {
    if (result == null) {
      return null;
    }

    // Priority order for validation messages
    if (result.confidence < 0.7) {
      return '문서나 책을 준비해 주세요';
    }

    if (result.segmentAreaRatio < 0.1) {
      return '문서를 가깝게 해주세요';
    }

    final cornerCount = result.corners?.length ?? 0;
    if (cornerCount < 4) {
      return '문서 전체를 보이게 해주세요';
    }

    if (cornerCount == 4 && !result.isParallel) {
      return '카메라를 평행하게 맞춰주세요';
    }

    return null;
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
            child: Stack(
              children: [
                // Full screen camera preview
                Positioned.fill(
                  child: _buildCameraPreview(),
                ),
                // Validation message overlay (center)
                if (_getValidationMessage(_latestResult) != null)
                  Center(
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 24,
                        vertical: 16,
                      ),
                      margin: const EdgeInsets.symmetric(horizontal: 32),
                      decoration: BoxDecoration(
                        color: Colors.black.withOpacity(0.7),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Text(
                        _getValidationMessage(_latestResult)!,
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 18,
                          fontWeight: FontWeight.w600,
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ),
                  ),
                // Info overlay (top right, small)
                Positioned(
                  top: 8,
                  right: 8,
                  child: _buildInfoOverlay(),
                ),
                // Bottom control panel
                Positioned(
                  bottom: 0,
                  left: 0,
                  right: 0,
                  child: _buildControlPanel(),
                ),
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
        height: double.infinity,
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
      height: double.infinity,
      color: Colors.black,
      child: Center(
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
      ),
    );
  }

  Widget _buildInfoOverlay() {
    final result = _latestResult;
    if (result == null) {
      return const SizedBox.shrink();
    }

    return Container(
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.6),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          Text(
            '${(result.confidence * 100).toStringAsFixed(0)}%',
            style: const TextStyle(
              color: Colors.white,
              fontSize: 12,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 2),
          Text(
            '${result.totalTime.inMilliseconds}ms',
            style: const TextStyle(
              color: Colors.white70,
              fontSize: 10,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildControlPanel() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.8),
        borderRadius: const BorderRadius.vertical(top: Radius.circular(20)),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // Top icon buttons
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              _buildIconButton(
                icon: _isTwoPageMode ? Icons.book : Icons.description,
                label: _isTwoPageMode ? '2페이지' : '1페이지',
                onPressed: () {
                  setState(() {
                    _isTwoPageMode = !_isTwoPageMode;
                  });
                },
              ),
              _buildIconButton(
                icon: _isFlashOn ? Icons.flash_on : Icons.flash_off,
                label: '플래시',
                onPressed: _toggleFlash,
              ),
              _buildIconButton(
                icon: Icons.settings,
                label: '설정',
                onPressed: () {
                  // TODO: Open settings
                },
              ),
            ],
          ),
          const SizedBox(height: 16),
          // Middle row: Capture button and folder selector
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // Folder selector
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.2),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: DropdownButton<String>(
                  value: _selectedFolder,
                  dropdownColor: Colors.grey.shade800,
                  underline: const SizedBox.shrink(),
                  style: const TextStyle(color: Colors.white, fontSize: 14),
                  items: _folders.map((folder) {
                    return DropdownMenuItem(
                      value: folder,
                      child: Text(folder),
                    );
                  }).toList(),
                  onChanged: (value) {
                    if (value != null) {
                      setState(() {
                        _selectedFolder = value;
                      });
                    }
                  },
                ),
              ),
              const SizedBox(width: 16),
              // Capture button
              GestureDetector(
                onTap: _captureImage,
                child: Container(
                  width: 70,
                  height: 70,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: Colors.white,
                    border: Border.all(
                      color: Colors.teal,
                      width: 4,
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.teal.withOpacity(0.5),
                        blurRadius: 10,
                        spreadRadius: 2,
                      ),
                    ],
                  ),
                  child: const Center(
                    child: Icon(
                      Icons.camera_alt,
                      color: Colors.teal,
                      size: 32,
                    ),
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildIconButton({
    required IconData icon,
    required String label,
    required VoidCallback onPressed,
  }) {
    return InkWell(
      onTap: onPressed,
      borderRadius: BorderRadius.circular(8),
      child: Padding(
        padding: const EdgeInsets.all(8),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, color: Colors.white, size: 28),
            const SizedBox(height: 4),
            Text(
              label,
              style: const TextStyle(
                color: Colors.white70,
                fontSize: 12,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _toggleFlash() async {
    final controller = _cameraController;
    if (controller == null || !controller.value.isInitialized) {
      return;
    }

    try {
      await controller.setFlashMode(
        _isFlashOn ? FlashMode.off : FlashMode.torch,
      );
      setState(() {
        _isFlashOn = !_isFlashOn;
      });
    } catch (e) {
      setState(() {
        _error = 'Failed to toggle flash: $e';
      });
    }
  }

  Future<void> _captureImage() async {
    final controller = _cameraController;
    final result = _latestResult;

    if (controller == null || !controller.value.isInitialized) {
      return;
    }

    if (result == null || result.corners == null || result.corners!.length < 4) {
      _showSnackBar('문서가 제대로 인식되지 않았습니다.');
      return;
    }

    try {
      // 1. Stop image stream
      await _stopImageStream();

      // 2. Take picture
      final xfile = await controller.takePicture();
      final imageBytes = await xfile.readAsBytes();
      final capturedImage = img.decodeImage(imageBytes);

      if (capturedImage == null) {
        throw StateError('Failed to decode captured image');
      }

      // 3. Apply orientation
      final oriented = _applyOrientation(capturedImage);

      // 4. Crop image using corners
      final croppedImage = _cropImageWithPerspective(oriented, result.corners!);

      // 5. Save to selected folder
      await _saveImage(croppedImage, _selectedFolder);

      _showSnackBar('이미지가 저장되었습니다.');

      // 6. Resume image stream
      await _startImageStream();
    } catch (e) {
      _showSnackBar('이미지 캡쳐 실패: $e');
      await _startImageStream();
    }
  }

  img.Image _cropImageWithPerspective(
    img.Image source,
    List<Corner> corners,
  ) {
    // For simplicity, we'll use a bounding box approach
    // In a production app, you'd use perspective transform
    final minX = corners.map((c) => c.x).reduce((a, b) => a < b ? a : b).toInt();
    final minY = corners.map((c) => c.y).reduce((a, b) => a < b ? a : b).toInt();
    final maxX = corners.map((c) => c.x).reduce((a, b) => a > b ? a : b).toInt();
    final maxY = corners.map((c) => c.y).reduce((a, b) => a > b ? a : b).toInt();

    final width = (maxX - minX).clamp(1, source.width - minX);
    final height = (maxY - minY).clamp(1, source.height - minY);

    return img.copyCrop(
      source,
      x: minX.clamp(0, source.width - 1),
      y: minY.clamp(0, source.height - 1),
      width: width,
      height: height,
    );
  }

  Future<void> _saveImage(img.Image image, String folderName) async {
    final directory = await getApplicationDocumentsDirectory();
    final folderPath = '${directory.path}/ml_scanner/$folderName';
    final folder = Directory(folderPath);

    if (!await folder.exists()) {
      await folder.create(recursive: true);
    }

    final timestamp = DateTime.now().millisecondsSinceEpoch;
    final fileName = 'scan_$timestamp.png';
    final filePath = '$folderPath/$fileName';

    final pngBytes = img.encodePng(image);
    final file = File(filePath);
    await file.writeAsBytes(pngBytes);
  }

  void _showSnackBar(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        duration: const Duration(seconds: 2),
      ),
    );
  }

  Future<void> _loadFolders() async {
    final directory = await getApplicationDocumentsDirectory();
    final basePath = '${directory.path}/ml_scanner';
    final baseDir = Directory(basePath);

    if (!await baseDir.exists()) {
      await baseDir.create(recursive: true);
      // Create default folder
      final defaultFolder = Directory('$basePath/기본');
      await defaultFolder.create();
      return;
    }

    final entities = await baseDir.list().toList();
    final folderNames = entities
        .whereType<Directory>()
        .map((dir) => dir.path.split('/').last)
        .toList();

    if (folderNames.isEmpty) {
      final defaultFolder = Directory('$basePath/기본');
      await defaultFolder.create();
      folderNames.add('기본');
    }

    setState(() {
      _folders.clear();
      _folders.addAll(folderNames);
      if (!_folders.contains(_selectedFolder)) {
        _selectedFolder = _folders.first;
      }
    });
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
