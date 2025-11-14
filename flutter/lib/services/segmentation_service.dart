import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';

class SegmentationResult {
  const SegmentationResult({
    required this.overlayBytes,
    required this.maskBytes,
    required this.maskWidth,
    required this.maskHeight,
    required this.totalTime,
    required this.preprocessTime,
    required this.inferenceTime,
    required this.postprocessTime,
    required this.confidence,
    required this.accelerated,
  });

  final Uint8List overlayBytes;
  final Uint8List maskBytes;
  final int maskWidth;
  final int maskHeight;
  final Duration totalTime;
  final Duration preprocessTime;
  final Duration inferenceTime;
  final Duration postprocessTime;
  final double confidence;
  final bool accelerated;
}

class SegmentationService {
  static const _modelAssetPath =
      'assets/models/deeplabv3plus_mobilenet_512.onnx';
  static const int _inputSize = 512;
  static const double _overlayAlpha = 0.45;
  static const _mean = [0.485, 0.456, 0.406];
  static const _std = [0.229, 0.224, 0.225];

  OrtSession? _session;
  OrtSessionOptions? _sessionOptions;
  OrtRunOptions? _runOptions;
  String? _inputName;
  bool _initialized = false;
  bool _usingAccelerated = false;

  Future<void> initialize() async {
    if (_initialized) {
      return;
    }

    OrtEnv.instance.init();
    _sessionOptions = OrtSessionOptions()
      ..setIntraOpNumThreads(2)
      ..setSessionGraphOptimizationLevel(GraphOptimizationLevel.ortEnableAll);

    _usingAccelerated = _tryEnableAcceleration();

    final rawModel = await rootBundle.load(_modelAssetPath);
    _session = OrtSession.fromBuffer(
      rawModel.buffer.asUint8List(),
      _sessionOptions!,
    );
    _inputName = _session!.inputNames.first;
    _runOptions = OrtRunOptions();
    _initialized = true;
  }

  Future<SegmentationResult> segment(Uint8List imageBytes) async {
    if (!_initialized) {
      await initialize();
    }
    final session = _session;
    final runOptions = _runOptions;
    final inputName = _inputName;
    if (session == null || runOptions == null || inputName == null) {
      throw StateError('Segmentation session is not ready.');
    }

    final totalStart = DateTime.now();
    final preprocessStart = DateTime.now();
    final originalImage = img.decodeImage(imageBytes);
    if (originalImage == null) {
      throw StateError('Failed to decode captured image.');
    }

    final resized = img.copyResize(
      originalImage,
      width: _inputSize,
      height: _inputSize,
      interpolation: img.Interpolation.linear,
    );

    final inputTensor = _buildInputTensor(resized);
    final inputs = <String, OrtValue>{inputName: inputTensor};
    final preprocessTime = DateTime.now().difference(preprocessStart);

    final inferenceStart = DateTime.now();
    final outputs = session.run(runOptions, inputs);
    final outputTensor = outputs.first as OrtValueTensor;
    final logits = outputTensor.value as List<dynamic>;
    final inferenceTime = DateTime.now().difference(inferenceStart);

    final postprocessStart = DateTime.now();
    final maskResult = _maskFromLogits(logits, _inputSize, _inputSize);
    final maskImage = _maskToImage(maskResult.mask, _inputSize, _inputSize);
    final resizedMask = img.copyResize(
      maskImage,
      width: originalImage.width,
      height: originalImage.height,
      interpolation: img.Interpolation.nearest,
    );
    final overlay = _overlayMask(originalImage, resizedMask);

    inputTensor.release();
    for (final output in outputs) {
      output?.release();
    }

    final postprocessTime = DateTime.now().difference(postprocessStart);
    final totalTime = DateTime.now().difference(totalStart);
    return SegmentationResult(
      overlayBytes: _encodeImage(overlay),
      maskBytes: _encodeImage(resizedMask),
      maskWidth: originalImage.width,
      maskHeight: originalImage.height,
      totalTime: totalTime,
      preprocessTime: preprocessTime,
      inferenceTime: inferenceTime,
      postprocessTime: postprocessTime,
      confidence: maskResult.confidence,
      accelerated: _usingAccelerated,
    );
  }

  void dispose() {
    _session?.release();
    _session = null;
    _runOptions?.release();
    _runOptions = null;
    _sessionOptions?.release();
    _sessionOptions = null;
    OrtEnv.instance.release();
    _initialized = false;
    _usingAccelerated = false;
  }

  static OrtValueTensor _buildInputTensor(img.Image image) {
    final totalPixels = image.width * image.height;
    final buffer = Float32List(totalPixels * 3);
    final redOffset = 0;
    final greenOffset = totalPixels;
    final blueOffset = totalPixels * 2;

    for (var y = 0; y < image.height; y++) {
      for (var x = 0; x < image.width; x++) {
        final pixel = image.getPixel(x, y);
        final index = y * image.width + x;
        final r = pixel.r.toDouble() / 255.0;
        final g = pixel.g.toDouble() / 255.0;
        final b = pixel.b.toDouble() / 255.0;
        buffer[redOffset + index] = (r - _mean[0]) / _std[0];
        buffer[greenOffset + index] = (g - _mean[1]) / _std[1];
        buffer[blueOffset + index] = (b - _mean[2]) / _std[2];
      }
    }

    return OrtValueTensor.createTensorWithDataList(buffer, [
      1,
      3,
      image.height,
      image.width,
    ]);
  }

  static _MaskResult _maskFromLogits(
    List<dynamic> logits,
    int width,
    int height,
  ) {
    if (logits.isEmpty) {
      return _MaskResult(List<int>.filled(width * height, 0), 0);
    }

    final batch = logits.first as List<dynamic>;
    if (batch.isEmpty) {
      return _MaskResult(List<int>.filled(width * height, 0), 0);
    }

    final classCount = batch.length;
    final mask = List<int>.filled(width * height, 0);
    var confidenceSum = 0.0;
    final pixelCount = width * height;

    if (classCount == 1) {
      final channel = batch.first as List<dynamic>;
      for (var y = 0; y < height; y++) {
        final row = channel[y] as List<dynamic>;
        for (var x = 0; x < width; x++) {
          final value = (row[x] as num).toDouble();
          final prob = _sigmoid(value);
          final predictedPositive = prob >= 0.5;
          mask[y * width + x] = predictedPositive ? 1 : 0;
          confidenceSum += predictedPositive ? prob : (1 - prob);
        }
      }
      final confidence = pixelCount == 0 ? 0 : confidenceSum / pixelCount;
      return _MaskResult(mask, confidence);
    }

    final scores = List<double>.filled(classCount, 0);
    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        var bestScore = double.negativeInfinity;
        var bestClass = 0;
        for (var c = 0; c < classCount; c++) {
          final channel = batch[c] as List<dynamic>;
          final row = channel[y] as List<dynamic>;
          final score = (row[x] as num).toDouble();
          scores[c] = score;
          if (score > bestScore) {
            bestScore = score;
            bestClass = c;
          }
        }

        var expSum = 0.0;
        for (var c = 0; c < classCount; c++) {
          final expScore = math.exp(scores[c] - bestScore);
          scores[c] = expScore;
          expSum += expScore;
        }
        final bestProb = expSum == 0 ? 0 : scores[bestClass] / expSum;
        confidenceSum += bestProb;
        mask[y * width + x] = bestClass == 1 ? 1 : 0;
      }
    }

    final confidence = pixelCount == 0 ? 0 : confidenceSum / pixelCount;
    return _MaskResult(mask, confidence);
  }

  static img.Image _maskToImage(List<int> mask, int width, int height) {
    final maskImage = img.Image(width: width, height: height);
    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        final value = mask[y * width + x] > 0 ? 255 : 0;
        maskImage.setPixelRgba(x, y, value, value, value, 255);
      }
    }
    return maskImage;
  }

  img.Image _overlayMask(img.Image original, img.Image mask) {
    final overlay = img.Image.from(original);
    for (var y = 0; y < overlay.height; y++) {
      for (var x = 0; x < overlay.width; x++) {
        final maskPixel = mask.getPixel(x, y);
        final maskValue = maskPixel.r.toInt();
        if (maskValue == 0) {
          continue;
        }
        final pixel = overlay.getPixel(x, y);
        final r = pixel.r.toDouble();
        final g = pixel.g.toDouble();
        final b = pixel.b.toDouble();
        final a = pixel.a.toInt();
        final blendedR = ((1 - _overlayAlpha) * r).round();
        final blendedG = ((1 - _overlayAlpha) * g + _overlayAlpha * 255.0)
            .round();
        final blendedB = ((1 - _overlayAlpha) * b).round();
        overlay.setPixelRgba(x, y, blendedR, blendedG, blendedB, a);
      }
    }
    return overlay;
  }

  static Uint8List _encodeImage(img.Image image) {
    return Uint8List.fromList(img.encodePng(image));
  }

  static double _sigmoid(double value) {
    return 1 / (1 + math.exp(-value));
  }

  bool _tryEnableAcceleration() {
    final options = _sessionOptions;
    if (options == null) {
      return false;
    }
    try {
      if (Platform.isAndroid) {
        return options.appendNnapiProvider(NnapiFlags.useFp16);
      }
      if (Platform.isIOS || Platform.isMacOS) {
        return options.appendCoreMLProvider(CoreMLFlags.useNone);
      }
    } catch (_) {
      return false;
    }
    return false;
  }
}

class _MaskResult {
  const _MaskResult(this.mask, this.confidence);

  final List<int> mask;
  final double confidence;
}
