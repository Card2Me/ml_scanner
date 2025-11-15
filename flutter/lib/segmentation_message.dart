import 'dart:typed_data';

class SegmentationRequest {
  const SegmentationRequest({
    required this.imageBytes,
  });

  final Uint8List imageBytes;
}

class SegmentationResponse {
  const SegmentationResponse({
    required this.overlayBytes,
    required this.maskBytes,
    required this.maskWidth,
    required this.maskHeight,
    required this.totalTime,
    required this.preprocessTime,
    required this.inferenceTime,
    required this.postprocessTime,
    required this.confidence,
    required this.corners,
  });

  final Uint8List overlayBytes;
  final Uint8List maskBytes;
  final int maskWidth;
  final int maskHeight;
  final int totalTime;
  final int preprocessTime;
  final int inferenceTime;
  final int postprocessTime;
  final double confidence;
  final List<double>? corners;
}
