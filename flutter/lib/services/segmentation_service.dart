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
    required this.segmentAreaRatio,
    required this.isParallel,
    this.corners,
    this.polygon,
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
  final double segmentAreaRatio; // 0-1, segment가 차지하는 비율
  final bool isParallel; // 외곽선이 평행한지 여부
  final List<Corner>? corners; // 4점 사각형 근사
  final List<Corner>? polygon; // 원래 다각형 (convex hull)
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
    final points = _extractSignificantPoints(maskResult.mask, _inputSize, _inputSize);
    final hull = points != null && points.isNotEmpty ? _convexHull(points) : null;
    final rawCorners = hull != null ? _approximateToQuadrilateral(hull) : null;
    final maskImage = _maskToImage(maskResult.mask, _inputSize, _inputSize);
    final resizedMask = img.copyResize(
      maskImage,
      width: originalImage.width,
      height: originalImage.height,
      interpolation: img.Interpolation.nearest,
    );
    final overlay = _overlayMask(originalImage, resizedMask, hull, rawCorners);

    // Scale both polygon and corners
    final scaleX = originalImage.width / _inputSize;
    final scaleY = originalImage.height / _inputSize;

    final scaledPolygon = hull?.map((point) {
      return Corner(
        point.x.toDouble(),
        point.y.toDouble(),
      ).scale(scaleX.toDouble(), scaleY.toDouble());
    }).toList();

    final scaledCorners = rawCorners?.map((point) {
      return Corner(
        point.x.toDouble(),
        point.y.toDouble(),
      ).scale(scaleX.toDouble(), scaleY.toDouble());
    }).toList();

    // Calculate segment area ratio
    final segmentAreaRatio = _calculateSegmentAreaRatio(maskResult.mask);

    // Check if corners form a parallel quadrilateral
    final isParallel = _checkParallelQuadrilateral(rawCorners);

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
      segmentAreaRatio: segmentAreaRatio,
      isParallel: isParallel,
      corners: scaledCorners,
      polygon: scaledPolygon,
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
    List<IntPoint>? polygon,
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

    final scaleX = original.width / _inputSize;
    final scaleY = original.height / _inputSize;

    // Draw polygon (convex hull) with weak/transparent line
    if (polygon != null && polygon.length >= 3) {
      final scaled = polygon.map((point) {
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
          color: img.ColorRgba8(255, 255, 0, 128), // Yellow, semi-transparent
          thickness: 2,
        );
      }
    }

    // Draw 4-point quadrilateral with strong line
    if (corners != null && corners.length == 4) {
      final scaled = corners.map((point) {
        final x = (point.x * scaleX).round();
        final y = (point.y * scaleY).round();
        return IntPoint(x, y);
      }).toList();
      for (var i = 0; i < scaled.length; i++) {
        final start = scaled[i];
        final end = scaled[(i + 1) % scaled.length];
        img.drawLine(
          overlay,
          x1: start.x,
          y1: start.y,
          x2: end.x,
          y2: end.y,
          color: img.ColorRgba8(0, 255, 0, 255), // Green, fully opaque
          thickness: 4,
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

  /// Approximates a polygon to a 4-point quadrilateral using Douglas-Peucker algorithm
  List<IntPoint>? _approximateToQuadrilateral(List<IntPoint> polygon) {
    if (polygon.length < 3) {
      return null;
    }
    if (polygon.length == 4) {
      return polygon;
    }

    // Use iterative Douglas-Peucker to reduce to 4 points
    var simplified = polygon;
    var epsilon = 1.0;
    const maxEpsilon = 100.0;
    const step = 0.5;

    while (simplified.length > 4 && epsilon < maxEpsilon) {
      simplified = _douglasPeucker(polygon, epsilon);
      epsilon += step;
    }

    // If we still have more than 4 points, select the 4 most significant ones
    if (simplified.length > 4) {
      simplified = _selectFourCorners(simplified);
    }

    // If we have less than 4 points, return null
    if (simplified.length < 4) {
      return null;
    }

    // Order corners: top-left, top-right, bottom-right, bottom-left
    return _orderCorners(simplified);
  }

  /// Douglas-Peucker polygon simplification algorithm
  List<IntPoint> _douglasPeucker(List<IntPoint> points, double epsilon) {
    if (points.length < 3) {
      return points;
    }

    // Find the point with maximum distance from line segment
    var maxDist = 0.0;
    var maxIndex = 0;
    final start = points.first;
    final end = points.last;

    for (var i = 1; i < points.length - 1; i++) {
      final dist = _perpendicularDistance(points[i], start, end);
      if (dist > maxDist) {
        maxDist = dist;
        maxIndex = i;
      }
    }

    // If max distance is greater than epsilon, recursively simplify
    if (maxDist > epsilon) {
      final left = _douglasPeucker(points.sublist(0, maxIndex + 1), epsilon);
      final right = _douglasPeucker(points.sublist(maxIndex), epsilon);
      return [...left.sublist(0, left.length - 1), ...right];
    } else {
      return [start, end];
    }
  }

  /// Calculate perpendicular distance from point to line segment
  double _perpendicularDistance(IntPoint point, IntPoint lineStart, IntPoint lineEnd) {
    final dx = lineEnd.x - lineStart.x;
    final dy = lineEnd.y - lineStart.y;
    final norm = math.sqrt(dx * dx + dy * dy);

    if (norm == 0) {
      final pdx = point.x - lineStart.x;
      final pdy = point.y - lineStart.y;
      return math.sqrt(pdx * pdx + pdy * pdy);
    }

    return ((point.y - lineStart.y) * dx - (point.x - lineStart.x) * dy).abs() / norm;
  }

  /// Select 4 corners from a polygon by finding the most distant points
  List<IntPoint> _selectFourCorners(List<IntPoint> points) {
    if (points.length <= 4) {
      return points;
    }

    // Find centroid
    var cx = 0.0;
    var cy = 0.0;
    for (final point in points) {
      cx += point.x;
      cy += point.y;
    }
    cx /= points.length;
    cy /= points.length;

    // Find 4 extreme points (top-left, top-right, bottom-right, bottom-left)
    IntPoint? topLeft, topRight, bottomRight, bottomLeft;
    var maxTL = double.negativeInfinity;
    var maxTR = double.negativeInfinity;
    var maxBR = double.negativeInfinity;
    var maxBL = double.negativeInfinity;

    for (final point in points) {
      final dx = point.x - cx;
      final dy = point.y - cy;

      // Top-left: minimize x + y
      final scoreTL = -(dx + dy);
      if (scoreTL > maxTL) {
        maxTL = scoreTL;
        topLeft = point;
      }

      // Top-right: maximize x - y
      final scoreTR = dx - dy;
      if (scoreTR > maxTR) {
        maxTR = scoreTR;
        topRight = point;
      }

      // Bottom-right: maximize x + y
      final scoreBR = dx + dy;
      if (scoreBR > maxBR) {
        maxBR = scoreBR;
        bottomRight = point;
      }

      // Bottom-left: minimize x - y
      final scoreBL = -(dx - dy);
      if (scoreBL > maxBL) {
        maxBL = scoreBL;
        bottomLeft = point;
      }
    }

    return [
      topLeft ?? points[0],
      topRight ?? points[1],
      bottomRight ?? points[2],
      bottomLeft ?? points[3],
    ];
  }

  /// Order corners in clockwise order starting from top-left
  List<IntPoint> _orderCorners(List<IntPoint> corners) {
    if (corners.length != 4) {
      return corners;
    }

    // Find centroid
    var cx = 0.0;
    var cy = 0.0;
    for (final corner in corners) {
      cx += corner.x;
      cy += corner.y;
    }
    cx /= 4;
    cy /= 4;

    // Sort by angle from centroid
    final sorted = List<IntPoint>.from(corners);
    sorted.sort((a, b) {
      final angleA = math.atan2(a.y - cy, a.x - cx);
      final angleB = math.atan2(b.y - cy, b.x - cx);
      return angleA.compareTo(angleB);
    });

    // Find top-left corner (minimum y, then minimum x)
    var topLeftIndex = 0;
    for (var i = 1; i < 4; i++) {
      if (sorted[i].y < sorted[topLeftIndex].y ||
          (sorted[i].y == sorted[topLeftIndex].y && sorted[i].x < sorted[topLeftIndex].x)) {
        topLeftIndex = i;
      }
    }

    // Rotate to start from top-left
    return [
      sorted[(topLeftIndex) % 4],
      sorted[(topLeftIndex + 1) % 4],
      sorted[(topLeftIndex + 2) % 4],
      sorted[(topLeftIndex + 3) % 4],
    ];
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

  static double _calculateSegmentAreaRatio(List<int> mask) {
    if (mask.isEmpty) {
      return 0.0;
    }
    final foregroundPixels = mask.where((pixel) => pixel > 0).length;
    return foregroundPixels / mask.length;
  }

  static bool _checkParallelQuadrilateral(List<IntPoint>? corners) {
    if (corners == null || corners.length != 4) {
      return false;
    }

    // Check if opposite sides are parallel
    // For a quadrilateral ABCD, check if AB is parallel to CD and BC is parallel to DA
    final threshold = 0.2; // Angle tolerance in radians (~11 degrees)

    // Calculate vectors for opposite sides
    final side1 = _vectorFrom(corners[0], corners[1]);
    final side2 = _vectorFrom(corners[1], corners[2]);
    final side3 = _vectorFrom(corners[2], corners[3]);
    final side4 = _vectorFrom(corners[3], corners[0]);

    // Check if side1 is parallel to side3
    final angle1 = _angleBetweenVectors(side1, side3);
    final parallel1 = angle1.abs() < threshold || (math.pi - angle1).abs() < threshold;

    // Check if side2 is parallel to side4
    final angle2 = _angleBetweenVectors(side2, side4);
    final parallel2 = angle2.abs() < threshold || (math.pi - angle2).abs() < threshold;

    return parallel1 && parallel2;
  }

  static List<double> _vectorFrom(IntPoint from, IntPoint to) {
    return [(to.x - from.x).toDouble(), (to.y - from.y).toDouble()];
  }

  static double _angleBetweenVectors(List<double> v1, List<double> v2) {
    final dotProduct = v1[0] * v2[0] + v1[1] * v2[1];
    final magnitude1 = math.sqrt(v1[0] * v1[0] + v1[1] * v1[1]);
    final magnitude2 = math.sqrt(v2[0] * v2[0] + v2[1] * v2[1]);

    if (magnitude1 == 0 || magnitude2 == 0) {
      return 0;
    }

    final cosAngle = dotProduct / (magnitude1 * magnitude2);
    return math.acos(cosAngle.clamp(-1.0, 1.0));
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

/// Perspective transform utility
class PerspectiveTransform {
  /// Apply perspective transform to warp an image based on 4 corner points
  /// Uses bilinear interpolation for mapping
  static img.Image? warpPerspective(
    img.Image source,
    List<Corner> corners, {
    int? outputWidth,
    int? outputHeight,
  }) {
    if (corners.length != 4) {
      return null;
    }

    // Order corners: top-left, top-right, bottom-right, bottom-left
    final ordered = _orderCornersForTransform(corners);

    // Calculate output dimensions if not provided
    final width = outputWidth ?? _calculateOutputWidth(ordered);
    final height = outputHeight ?? _calculateOutputHeight(ordered);

    if (width <= 0 || height <= 0) {
      return null;
    }

    // Create output image
    final output = img.Image(width: width, height: height);

    // Use bilinear mapping from destination to source
    // Each point in destination maps to corresponding point in source quadrilateral
    for (var dstY = 0; dstY < height; dstY++) {
      for (var dstX = 0; dstX < width; dstX++) {
        // Normalize coordinates to [0, 1]
        final u = dstX / width;
        final v = dstY / height;

        // Bilinear interpolation in source quadrilateral
        final srcX = _bilinearInterpolate(
          ordered[0].x, // top-left
          ordered[1].x, // top-right
          ordered[2].x, // bottom-right
          ordered[3].x, // bottom-left
          u,
          v,
        );

        final srcY = _bilinearInterpolate(
          ordered[0].y, // top-left
          ordered[1].y, // top-right
          ordered[2].y, // bottom-right
          ordered[3].y, // bottom-left
          u,
          v,
        );

        // Get pixel from source image with bilinear interpolation
        final pixel = _getPixelBilinear(source, srcX, srcY);
        if (pixel != null) {
          output.setPixel(dstX, dstY, pixel);
        }
      }
    }

    return output;
  }

  /// Bilinear interpolation for quadrilateral mapping
  /// tl, tr, br, bl are values at top-left, top-right, bottom-right, bottom-left
  /// u, v are normalized coordinates in [0, 1]
  static double _bilinearInterpolate(
    double tl,
    double tr,
    double br,
    double bl,
    double u,
    double v,
  ) {
    final top = tl * (1 - u) + tr * u;
    final bottom = bl * (1 - u) + br * u;
    return top * (1 - v) + bottom * v;
  }

  /// Get pixel value with bilinear interpolation
  static img.Pixel? _getPixelBilinear(img.Image image, double x, double y) {
    if (x < 0 || y < 0 || x >= image.width - 1 || y >= image.height - 1) {
      // Handle edge cases
      final ix = x.round().clamp(0, image.width - 1);
      final iy = y.round().clamp(0, image.height - 1);
      return image.getPixel(ix, iy);
    }

    final x0 = x.floor();
    final y0 = y.floor();
    final x1 = x0 + 1;
    final y1 = y0 + 1;

    final fx = x - x0;
    final fy = y - y0;

    final p00 = image.getPixel(x0, y0);
    final p10 = image.getPixel(x1, y0);
    final p01 = image.getPixel(x0, y1);
    final p11 = image.getPixel(x1, y1);

    final r = ((1 - fx) * (1 - fy) * p00.r +
            fx * (1 - fy) * p10.r +
            (1 - fx) * fy * p01.r +
            fx * fy * p11.r)
        .round();

    final g = ((1 - fx) * (1 - fy) * p00.g +
            fx * (1 - fy) * p10.g +
            (1 - fx) * fy * p01.g +
            fx * fy * p11.g)
        .round();

    final b = ((1 - fx) * (1 - fy) * p00.b +
            fx * (1 - fy) * p10.b +
            (1 - fx) * fy * p01.b +
            fx * fy * p11.b)
        .round();

    return img.ColorRgb8(r, g, b);
  }

  static List<Corner> _orderCornersForTransform(List<Corner> corners) {
    // Find centroid
    var cx = 0.0;
    var cy = 0.0;
    for (final corner in corners) {
      cx += corner.x;
      cy += corner.y;
    }
    cx /= 4;
    cy /= 4;

    // Classify corners
    Corner? topLeft, topRight, bottomRight, bottomLeft;

    for (final corner in corners) {
      if (corner.x < cx && corner.y < cy) {
        topLeft = corner;
      } else if (corner.x >= cx && corner.y < cy) {
        topRight = corner;
      } else if (corner.x >= cx && corner.y >= cy) {
        bottomRight = corner;
      } else {
        bottomLeft = corner;
      }
    }

    return [
      topLeft ?? corners[0],
      topRight ?? corners[1],
      bottomRight ?? corners[2],
      bottomLeft ?? corners[3],
    ];
  }

  static int _calculateOutputWidth(List<Corner> corners) {
    final topWidth = (corners[1].x - corners[0].x).abs();
    final bottomWidth = (corners[2].x - corners[3].x).abs();
    return math.max(topWidth, bottomWidth).round();
  }

  static int _calculateOutputHeight(List<Corner> corners) {
    final leftHeight = (corners[3].y - corners[0].y).abs();
    final rightHeight = (corners[2].y - corners[1].y).abs();
    return math.max(leftHeight, rightHeight).round();
  }
}
