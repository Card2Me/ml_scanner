import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';

class IntPoint {
  const IntPoint(this.x, this.y);

  final int x;
  final int y;
}

class Corner {
  const Corner(this.x, this.y);

  final double x;
  final double y;

  Corner scale(double scaleX, double scaleY) {
    return Corner(x * scaleX, y * scaleY);
  }
}

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
    this.corners,
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
  final List<Corner>? corners;
}

class SegmentationService {
  static const _modelAssetPath =
      'assets/models/deeplabv3plus_mobilenet_256.onnx';
  static const int _inputSize = 256;
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
    final originalImage = img.decodeImage(imageBytes);
    if (originalImage == null) {
      throw StateError('Failed to decode captured image.');
    }
    return segmentImage(originalImage);
  }

  Future<SegmentationResult> segmentImage(img.Image originalImage) async {
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

    final resized = img.copyResize(
      originalImage,
      width: _inputSize,
      height: _inputSize,
      interpolation: img.Interpolation.nearest,
    );

    final inputTensor = _buildInputTensor(resized);
    final inputs = <String, OrtValue>{inputName: inputTensor};
    final preprocessTime = DateTime.now().difference(preprocessStart);

    final inferenceStart = DateTime.now();
    final outputs = await session.runAsync(runOptions, inputs);
    final outputTensor = outputs!.first as OrtValueTensor;
    final logits = outputTensor.value as List<dynamic>;
    final inferenceTime = DateTime.now().difference(inferenceStart);

    final postprocessStart = DateTime.now();
    final maskResult = _maskFromLogits(logits, _inputSize, _inputSize);
    final rawCorners = _findCorners(maskResult.mask, _inputSize, _inputSize);
    final maskImage = _maskToImage(maskResult.mask, _inputSize, _inputSize);
    final resizedMask = img.copyResize(
      maskImage,
      width: originalImage.width,
      height: originalImage.height,
      interpolation: img.Interpolation.nearest,
    );
    final overlay = _overlayMask(originalImage, resizedMask, rawCorners);
    final scaledCorners = rawCorners?.map((point) {
      final scaleX = originalImage.width / _inputSize;
      final scaleY = originalImage.height / _inputSize;
      return Corner(
        point.x.toDouble(),
        point.y.toDouble(),
      ).scale(scaleX.toDouble(), scaleY.toDouble());
    }).toList();

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
      corners: scaledCorners,
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
      return _MaskResult(mask, confidence.toDouble());
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
    return _MaskResult(mask, confidence.toDouble());
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

  img.Image _overlayMask(
    img.Image original,
    img.Image mask,
    List<IntPoint>? corners,
  ) {
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

    if (corners != null && corners.length >= 3) {
      final scaleX = original.width / _inputSize;
      final scaleY = original.height / _inputSize;
      final scaled = corners.map((point) {
        final x = (point.x * scaleX).round();
        final y = (point.y * scaleY).round();
        return IntPoint(x, y);
      }).toList();
      final smoothed = _smoothPolyline(scaled);
      for (var i = 0; i < smoothed.length; i++) {
        final start = smoothed[i];
        final end = smoothed[(i + 1) % smoothed.length];
        img.drawLine(
          overlay,
          x1: start.x,
          y1: start.y,
          x2: end.x,
          y2: end.y,
          color: img.ColorRgba8(255, 0, 0, 255),
          thickness: 3,
        );
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

  List<IntPoint>? _findCorners(List<int> mask, int width, int height) {
    final points = _extractSignificantPoints(mask, width, height);
    if (points == null || points.isEmpty) {
      return null;
    }
    final hull = _convexHull(points);
    return hull;
  }

  List<IntPoint>? _extractSignificantPoints(
    List<int> mask,
    int width,
    int height,
  ) {
    final visited = List<bool>.filled(mask.length, false);
    final components = <List<IntPoint>>[];

    final directions = const [
      IntPoint(1, 0),
      IntPoint(-1, 0),
      IntPoint(0, 1),
      IntPoint(0, -1),
    ];

    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        final index = y * width + x;
        if (mask[index] == 0 || visited[index]) {
          continue;
        }
        final queue = <int>[index];
        final component = <IntPoint>[];
        visited[index] = true;
        while (queue.isNotEmpty) {
          final current = queue.removeLast();
          final cx = current % width;
          final cy = current ~/ width;
          component.add(IntPoint(cx, cy));
          for (final dir in directions) {
            final nx = cx + dir.x;
            final ny = cy + dir.y;
            if (nx < 0 || ny < 0 || nx >= width || ny >= height) {
              continue;
            }
            final nIndex = ny * width + nx;
            if (mask[nIndex] == 0 || visited[nIndex]) {
              continue;
            }
            visited[nIndex] = true;
            queue.add(nIndex);
          }
        }
        components.add(component);
      }
    }

    if (components.isEmpty) {
      return null;
    }

    if (components.length == 1) {
      return components.first;
    }

    components.sort((a, b) => a.length.compareTo(b.length));
    components.removeAt(0); // remove the smallest component

    final merged = <IntPoint>[];
    for (final component in components) {
      merged.addAll(component);
    }
    return merged;
  }

  List<IntPoint> _convexHull(List<IntPoint> points) {
    points.sort((a, b) {
      final dx = a.x - b.x;
      return dx != 0 ? dx : a.y - b.y;
    });
    final lower = <IntPoint>[];
    for (final p in points) {
      while (lower.length >= 2 &&
          _cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
        lower.removeLast();
      }
      lower.add(p);
    }
    final upper = <IntPoint>[];
    for (final p in points.reversed) {
      while (upper.length >= 2 &&
          _cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
        upper.removeLast();
      }
      upper.add(p);
    }
    lower.removeLast();
    upper.removeLast();
    return [...lower, ...upper];
  }

  int _cross(IntPoint a, IntPoint b, IntPoint c) {
    final abx = b.x - a.x;
    final aby = b.y - a.y;
    final acx = c.x - a.x;
    final acy = c.y - a.y;
    return abx * acy - aby * acx;
  }
}

class _MaskResult {
  const _MaskResult(this.mask, this.confidence);

  final List<int> mask;
  final double confidence;
}

List<IntPoint> _smoothPolyline(List<IntPoint> points) {
  if (points.length <= 2) {
    return points;
  }
  final smoothed = <IntPoint>[];
  for (var i = 0; i < points.length; i++) {
    final prev = points[(i - 1 + points.length) % points.length];
    final current = points[i];
    final next = points[(i + 1) % points.length];
    final avgX = ((prev.x + current.x + next.x) / 3).round();
    final avgY = ((prev.y + current.y + next.y) / 3).round();
    smoothed.add(IntPoint(avgX, avgY));
  }
  return smoothed;
}
