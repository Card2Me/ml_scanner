import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

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

class _ScannerScreenState extends State<ScannerScreen> {
  final SegmentationService _segmentationService = SegmentationService();
  CameraController? _cameraController;
  Future<void>? _initialization;
  SegmentationResult? _latestResult;
  bool _isProcessing = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    _initialization = _initialize();
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    _segmentationService.dispose();
    super.dispose();
  }

  Future<void> _initialize() async {
    final cameras = await availableCameras();
    if (cameras.isEmpty) {
      throw StateError('No camera available on this device');
    }

    final controller = CameraController(
      cameras.first,
      ResolutionPreset.high,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );
    _cameraController = controller;
    await controller.initialize();
    await _segmentationService.initialize();
  }

  Future<void> _captureAndSegment() async {
    final controller = _cameraController;
    if (controller == null || !controller.value.isInitialized) {
      return;
    }

    setState(() {
      _isProcessing = true;
      _error = null;
    });

    try {
      final capture = await controller.takePicture();
      final bytes = await capture.readAsBytes();
      final result = await _segmentationService.segment(bytes);
      if (!mounted) {
        return;
      }
      setState(() {
        _latestResult = result;
      });
    } catch (error) {
      if (!mounted) {
        return;
      }
      setState(() {
        _error = error.toString();
      });
    } finally {
      if (!mounted) {
        return;
      }
      setState(() {
        _isProcessing = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Document Segmenter')),
      body: FutureBuilder<void>(
        future: _initialization,
        builder: (context, snapshot) {
          if (snapshot.connectionState != ConnectionState.done) {
            return const Center(child: CircularProgressIndicator());
          }
          if (snapshot.hasError) {
            return _ErrorState(message: snapshot.error.toString());
          }
          return Column(
            children: [
              Expanded(child: _buildCameraPreview()),
              _buildResultPanel(),
            ],
          );
        },
      ),
    );
  }

  Widget _buildCameraPreview() {
    final controller = _cameraController;
    if (controller == null || !controller.value.isInitialized) {
      return const Center(child: Text('Camera is not ready.'));
    }

    return Stack(
      fit: StackFit.expand,
      children: [
        CameraPreview(controller),
        if (_latestResult != null)
          IgnorePointer(
            child: AnimatedOpacity(
              opacity: _isProcessing ? 0 : 0.85,
              duration: const Duration(milliseconds: 250),
              child: Image.memory(
                _latestResult!.overlayBytes,
                fit: BoxFit.cover,
              ),
            ),
          ),
        if (_latestResult != null)
          Positioned(
            top: 24,
            left: 24,
            child: _MetricsOverlay(
              result: _latestResult!,
              formatter: _formatDuration,
            ),
          ),
        if (_isProcessing)
          Container(
            color: Colors.black45,
            child: const Center(child: CircularProgressIndicator()),
          ),
        Positioned(
          bottom: 32,
          left: 0,
          right: 0,
          child: Center(
            child: FloatingActionButton.extended(
              onPressed: _isProcessing ? null : _captureAndSegment,
              icon: const Icon(Icons.camera_alt),
              label: const Text('Capture'),
            ),
          ),
        ),
      ],
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
              'Ready to scan',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 4),
            const Text('Tap the capture button to run segmentation.'),
            if (_error != null) ...[
              const SizedBox(height: 8),
              Text(_error!, style: const TextStyle(color: Colors.red)),
            ],
          ],
        ),
      );
    }

    final result = _latestResult!;
    return Container(
      width: double.infinity,
      color: Colors.grey.shade50,
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
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
          const SizedBox(height: 12),
          ClipRRect(
            borderRadius: BorderRadius.circular(12),
            child: SizedBox(
              height: 150,
              width: double.infinity,
              child: Image.memory(result.maskBytes, fit: BoxFit.cover),
            ),
          ),
          if (_error != null) ...[
            const SizedBox(height: 8),
            Text(_error!, style: const TextStyle(color: Colors.red)),
          ],
        ],
      ),
    );
  }

  String _formatDuration(Duration duration) {
    return '${duration.inMilliseconds} ms';
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

class _MetricsOverlay extends StatelessWidget {
  const _MetricsOverlay({required this.result, required this.formatter});

  final SegmentationResult result;
  final String Function(Duration) formatter;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.6),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            '확신도 ${(result.confidence * 100).toStringAsFixed(1)}%',
            style: const TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.bold,
            ),
          ),
          Text(
            result.accelerated ? '가속 모드 (NNAPI/CoreML)' : 'CPU 모드',
            style: const TextStyle(color: Colors.white70),
          ),
          const SizedBox(height: 4),
          Text(
            '전처리 ${formatter(result.preprocessTime)}',
            style: const TextStyle(color: Colors.white70),
          ),
          Text(
            '추론 ${formatter(result.inferenceTime)}',
            style: const TextStyle(color: Colors.white70),
          ),
          Text(
            '후처리 ${formatter(result.postprocessTime)}',
            style: const TextStyle(color: Colors.white70),
          ),
        ],
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
        _InfoChip(
          label: '가속',
          value: result.accelerated ? 'ON' : 'OFF',
        ),
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
