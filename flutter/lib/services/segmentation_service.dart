import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';

import '../utils/geometry_utils.dart';

/// 문서 세그멘테이션 결과를 담는 클래스
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

  final Uint8List overlayBytes; // 오버레이 이미지 바이트 (사용 안함)
  final Uint8List maskBytes; // 마스크 이미지 바이트
  final int maskWidth; // 마스크 너비
  final int maskHeight; // 마스크 높이
  final Duration totalTime; // 전체 처리 시간
  final Duration preprocessTime; // 전처리 시간
  final Duration inferenceTime; // 추론 시간
  final Duration postprocessTime; // 후처리 시간
  final double confidence; // 확신도 (0.0 ~ 1.0)
  final bool accelerated; // 하드웨어 가속 사용 여부
  final double segmentAreaRatio; // 세그먼트가 차지하는 영역 비율 (0.0 ~ 1.0)
  final bool isParallel; // 외곽선이 평행한지 여부
  final List<Corner>? corners; // 4점 사각형 근사 좌표
  final List<Corner>? polygon; // 원래 다각형 (convex hull)
}

/// 문서 세그멘테이션 서비스
/// ONNX 런타임을 사용하여 이미지에서 문서 영역을 탐지
class SegmentationService {
  static const _modelAssetPath =
      'assets/models/deeplabv3plus_mobilenet_256.onnx';
  static const int _inputSize = 256; // 모델 입력 크기
  static const _mean = [0.485, 0.456, 0.406]; // 정규화 평균값
  static const _std = [0.229, 0.224, 0.225]; // 정규화 표준편차

  OrtSession? _session; // ONNX 세션
  OrtSessionOptions? _sessionOptions; // 세션 옵션
  OrtRunOptions? _runOptions; // 실행 옵션
  String? _inputName; // 입력 텐서 이름
  bool _initialized = false; // 초기화 여부
  bool _usingAccelerated = false; // 하드웨어 가속 사용 여부

  /// ONNX 런타임 초기화
  /// 모델 로드 및 세션 설정
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

  /// 이미지 바이트로부터 세그멘테이션 수행
  Future<SegmentationResult> segment(Uint8List imageBytes) async {
    final originalImage = img.decodeImage(imageBytes);
    if (originalImage == null) {
      throw StateError('Failed to decode captured image.');
    }
    return segmentImage(originalImage);
  }

  /// 이미지 객체로부터 세그멘테이션 수행
  /// 문서 영역을 탐지하고 꼭지점 좌표를 반환
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
    final points = GeometryUtils.extractSignificantPoints(
      maskResult.mask,
      _inputSize,
      _inputSize,
    );
    final hull = points != null && points.isNotEmpty
        ? GeometryUtils.convexHull(points)
        : null;
    final rawCorners = hull != null
        ? GeometryUtils.approximateToQuadrilateral(hull)
        : null;
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
    final segmentAreaRatio = GeometryUtils.calculateSegmentAreaRatio(
      maskResult.mask,
    );

    // Check if corners form a parallel quadrilateral
    final isParallel = GeometryUtils.checkParallelQuadrilateral(rawCorners);

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

  /// 리소스 해제
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

  /// 이미지를 ONNX 입력 텐서로 변환
  /// RGB 채널을 정규화하여 Float32 텐서 생성
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

  /// 모델 출력 로짓으로부터 마스크 생성
  /// 소프트맥스를 적용하여 이진 마스크와 확신도 계산
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

  /// 마스크 배열을 이미지로 변환
  /// 1은 흰색(255), 0은 검은색(0)으로 매핑
  static img.Image _maskToImage(List<int> mask, int width, int height) {
    final maskImage = img.Image(width: width, height: height, numChannels: 4);
    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        final isForeground = mask[y * width + x] > 0;
        if (isForeground) {
          // Green with ~20% opacity (50/255) for subtle highlight
          maskImage.setPixelRgba(x, y, 0, 255, 0, 50);
        } else {
          // Transparent
          maskImage.setPixelRgba(x, y, 0, 0, 0, 0);
        }
      }
    }
    return maskImage;
  }

  /// 마스크를 원본 이미지에 오버레이 (사용 안함 - 그림자 제거)
  /// 빈 이미지를 반환하도록 수정
  img.Image _overlayMask(
    img.Image original,
    img.Image mask,
    List<IntPoint>? polygon,
    List<IntPoint>? corners,
  ) {
    // 그림자 효과 제거를 위해 오버레이 없이 빈 이미지 반환
    final overlay = img.Image.from(original);
    return overlay;
  }

  /// 이미지를 PNG 바이트로 인코딩
  static Uint8List _encodeImage(img.Image image) {
    return Uint8List.fromList(img.encodePng(image));
  }

  /// 시그모이드 함수
  /// 로짓 값을 0~1 확률로 변환
  static double _sigmoid(double value) {
    return 1 / (1 + math.exp(-value));
  }

  /// 하드웨어 가속 활성화 시도
  /// Android는 NNAPI, iOS/macOS는 CoreML 사용
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

/// 마스크 결과와 확신도를 담는 내부 클래스
class _MaskResult {
  const _MaskResult(this.mask, this.confidence);

  final List<int> mask; // 이진 마스크
  final double confidence; // 확신도
}
