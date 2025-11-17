import 'dart:async';
import 'dart:io';

import 'dart:ui' as ui;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';

import 'screens/folder_manager_screen.dart';
import 'screens/image_detail_screen.dart';
import 'services/segmentation_service.dart';
import 'theme/app_theme.dart';

/// 앱 진입점
Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MlScannerApp());
}

/// 메인 앱 위젯
class MlScannerApp extends StatelessWidget {
  const MlScannerApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Document Scanner',
      theme: AppTheme.lightTheme,
      darkTheme: AppTheme.darkTheme,
      debugShowCheckedModeBanner: false,
      home: const ScannerScreen(),
    );
  }
}

/// 스캐너 메인 화면
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
  late final AnimationController _cornerController;
  List<Offset>? _startCorners;
  List<Offset>? _targetCorners;

  // UI 상태 변수
  bool _isFlashOn = false; // 플래시 상태
  bool _isTwoPageMode = false; // 2페이지 스캔 모드
  String _selectedFolder = '기본'; // 선택된 폴더
  final Map<String, int> _folderFileCounts = {}; // 폴더별 파일 개수
  String? _lastCapturedImagePath; // 마지막 캡처된 이미지 경로
  int _lastCapturedFolderFileCount = 0; // 마지막 캡처된 폴더의 파일 개수

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

  /// 카메라 및 세그멘테이션 서비스 초기화
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

  /// 카메라 이미지 스트림 시작
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

  /// 카메라 이미지 스트림 중지
  Future<void> _stopImageStream() async {
    final controller = _cameraController;
    if (controller == null || !_isStreaming) {
      return;
    }
    await controller.stopImageStream();
    _isStreaming = false;
  }

  /// 카메라 이미지 처리 및 세그멘테이션 수행
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

  /// 세그멘테이션 결과 업데이트
  void _updateResult(SegmentationResult result) {
    setState(() {
      _latestResult = result;
      _error = null;
    });
    _updateCorners(result);
  }

  /// 유효성 검증 메시지 반환
  /// 확신도, 영역 비율, 코너 개수, 평행 여부 등을 체크
  String? _getValidationMessage(SegmentationResult? result) {
    if (result == null) {
      return null;
    }

    // 확신도 85% 미만일 때
    if (result.confidence < 0.85) {
      return '문서를 찾을 수 없습니다';
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

  /// 캡처 가능 여부 확인
  /// 확신도 85% 이상이고 유효성 검사를 통과해야 함
  bool _canCapture() {
    final result = _latestResult;
    if (result == null) return false;

    // 확신도 85% 이상이고 유효성 메시지가 없을 때 캡처 가능
    return result.confidence >= 0.85 && _getValidationMessage(result) == null;
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
                // 전체 화면 카메라 프리뷰
                Positioned.fill(child: _buildCameraPreview()),
                // 유효성 검증 메시지 오버레이 (중앙)
                if (_getValidationMessage(_latestResult) != null)
                  Center(
                    child: Container(
                      padding: const EdgeInsets.all(20),
                      margin: const EdgeInsets.symmetric(horizontal: 32),
                      decoration: BoxDecoration(
                        gradient: LinearGradient(
                          colors: [
                            AppTheme.warningColor.withOpacity(0.95),
                            AppTheme.warningColor.withOpacity(0.9),
                          ],
                          begin: Alignment.topLeft,
                          end: Alignment.bottomRight,
                        ),
                        borderRadius: BorderRadius.circular(20),
                        boxShadow: [
                          BoxShadow(
                            color: AppTheme.warningColor.withOpacity(0.4),
                            blurRadius: 20,
                            offset: const Offset(0, 8),
                          ),
                        ],
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          const Icon(
                            Icons.info_outline,
                            color: AppTheme.textPrimary,
                            size: 28,
                          ),
                          const SizedBox(width: 12),
                          Flexible(
                            child: Text(
                              _getValidationMessage(_latestResult)!,
                              style: const TextStyle(
                                color: AppTheme.textPrimary,
                                fontSize: 16,
                                fontWeight: FontWeight.w600,
                              ),
                              textAlign: TextAlign.center,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                // 정보 오버레이 (우측 상단, 작게)
                Positioned(top: 8, right: 8, child: _buildInfoOverlay()),
                // 하단 컨트롤 패널
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

  /// 카메라 프리뷰 위젯 빌드
  /// 문서 외곽선 오버레이 포함
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
      child: Align(
        alignment: Alignment.topCenter,
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
                      child: CustomPaint(
                        painter: _OutlinePainter(
                          polygon: _latestResult!.polygon,
                          corners: _latestResult!.corners,
                          maskWidth: _latestResult!.maskWidth.toDouble(),
                          maskHeight: _latestResult!.maskHeight.toDouble(),
                          confidence: _latestResult!.confidence,
                        ),
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

  /// 정보 오버레이 빌드 (확신도, 처리 시간)
  Widget _buildInfoOverlay() {
    final result = _latestResult;
    if (result == null) {
      return const SizedBox.shrink();
    }

    final isHighConfidence = result.confidence >= 0.85;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        gradient: isHighConfidence
            ? LinearGradient(
                colors: [
                  AppTheme.successColor.withOpacity(0.9),
                  AppTheme.successColor.withOpacity(0.8),
                ],
              )
            : LinearGradient(
                colors: [
                  Colors.black.withOpacity(0.7),
                  Colors.black.withOpacity(0.6),
                ],
              ),
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.2),
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(
                isHighConfidence ? Icons.check_circle : Icons.info,
                color: Colors.white,
                size: 14,
              ),
              const SizedBox(width: 4),
              Text(
                '${(result.confidence * 100).toStringAsFixed(0)}%',
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 14,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          const SizedBox(height: 2),
          Text(
            '${result.totalTime.inMilliseconds}ms',
            style: TextStyle(
              color: Colors.white.withOpacity(0.8),
              fontSize: 10,
            ),
          ),
        ],
      ),
    );
  }

  /// 하단 컨트롤 패널 빌드
  Widget _buildControlPanel() {
    final canCapture = _canCapture();

    return Container(
      padding: const EdgeInsets.only(left: 20, right: 20, top: 20, bottom: 32),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: const BorderRadius.vertical(top: Radius.circular(32)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.1),
            blurRadius: 20,
            offset: const Offset(0, -4),
          ),
        ],
      ),
      child: SafeArea(
        top: false,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // 상단 메뉴 바
            Container(
              padding: const EdgeInsets.symmetric(vertical: 8),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  _buildMenuButton(
                    icon: _isTwoPageMode ? Icons.menu_book : Icons.description,
                    label: _isTwoPageMode ? '2페이지' : '1페이지',
                    isActive: _isTwoPageMode,
                    onPressed: () {
                      setState(() {
                        _isTwoPageMode = !_isTwoPageMode;
                      });
                    },
                  ),
                  _buildMenuButton(
                    icon: _isFlashOn ? Icons.flash_on : Icons.flash_off,
                    label: '플래시',
                    isActive: _isFlashOn,
                    onPressed: _toggleFlash,
                  ),
                  _buildMenuButton(
                    icon: Icons.settings,
                    label: '설정',
                    isActive: false,
                    onPressed: () {
                      // TODO: 설정 화면 열기
                    },
                  ),
                ],
              ),
            ),
            const SizedBox(height: 20),
            // 중간 행: 폴더 선택, 캡처 버튼, 썸네일
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                // 폴더 선택 버튼
                Expanded(child: _buildFolderSelector()),
                const SizedBox(width: 12),
                // 캡처 버튼 (중앙)
                _buildCaptureButton(canCapture),
                const SizedBox(width: 12),
                // 썸네일 (또는 빈 공간)
                _buildThumbnail(),
              ],
            ),
          ],
        ),
      ),
    );
  }

  /// 폴더 선택 드롭다운 빌드
  Widget _buildFolderSelector() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: AppTheme.surfaceColor,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.grey.shade200),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.04),
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Icon(Icons.folder, color: AppTheme.primaryColor, size: 20),
          const SizedBox(width: 8),
          Flexible(
            child: DropdownButton<String>(
              value: _selectedFolder,
              dropdownColor: AppTheme.surfaceColor,
              underline: const SizedBox.shrink(),
              isDense: true,
              style: const TextStyle(
                color: AppTheme.textPrimary,
                fontSize: 14,
                fontWeight: FontWeight.w600,
              ),
              icon: const Icon(Icons.arrow_drop_down, color: AppTheme.textSecondary),
              items: [
                ..._folderFileCounts.keys.map((folder) {
                  final count = _folderFileCounts[folder] ?? 0;
                  return DropdownMenuItem(
                    value: folder,
                    child: Text(
                      '$folder ($count)',
                      overflow: TextOverflow.ellipsis,
                    ),
                  );
                }),
                const DropdownMenuItem(
                  value: '__manage__',
                  child: Row(
                    children: [
                      Icon(Icons.settings, size: 16, color: AppTheme.primaryColor),
                      SizedBox(width: 8),
                      Text('폴더 관리...'),
                    ],
                  ),
                ),
              ],
              onChanged: (value) async {
                if (value == '__manage__') {
                  // 폴더 관리 화면 열기
                  await Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => const FolderManagerScreen(),
                    ),
                  );
                  // 돌아왔을 때 폴더 목록 새로고침
                  await _loadFolders();
                } else if (value != null) {
                  setState(() {
                    _selectedFolder = value;
                  });
                }
              },
            ),
          ),
        ],
      ),
    );
  }

  /// 캡처 버튼 빌드
  /// 확신도 85% 이상일 때만 활성화
  Widget _buildCaptureButton(bool enabled) {
    return GestureDetector(
      onTap: enabled ? _captureImage : null,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 300),
        width: 70,
        height: 70,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          gradient: enabled ? AppTheme.primaryGradient : null,
          color: enabled ? null : Colors.grey.shade300,
          boxShadow: enabled
              ? [
                  BoxShadow(
                    color: AppTheme.primaryColor.withOpacity(0.4),
                    blurRadius: 20,
                    spreadRadius: 2,
                    offset: const Offset(0, 8),
                  ),
                ]
              : [],
        ),
        child: Center(
          child: Icon(
            Icons.camera_alt_rounded,
            color: enabled ? Colors.white : Colors.grey.shade400,
            size: 32,
          ),
        ),
      ),
    );
  }

  /// 메뉴 버튼 빌드
  Widget _buildMenuButton({
    required IconData icon,
    required String label,
    required VoidCallback onPressed,
    required bool isActive,
  }) {
    return InkWell(
      onTap: onPressed,
      borderRadius: BorderRadius.circular(12),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
        decoration: BoxDecoration(
          color: isActive
              ? AppTheme.primaryColor.withOpacity(0.1)
              : Colors.transparent,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              icon,
              color: isActive ? AppTheme.primaryColor : AppTheme.textSecondary,
              size: 24,
            ),
            const SizedBox(height: 4),
            Text(
              label,
              style: TextStyle(
                color: isActive ? AppTheme.primaryColor : AppTheme.textSecondary,
                fontSize: 11,
                fontWeight: isActive ? FontWeight.w600 : FontWeight.normal,
              ),
            ),
          ],
        ),
      ),
    );
  }

  /// 썸네일 빌드
  Widget _buildThumbnail() {
    if (_lastCapturedImagePath == null) {
      return Container(
        width: 56,
        height: 56,
        decoration: BoxDecoration(
          color: Colors.grey.shade100,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: Colors.grey.shade200),
        ),
        child: Icon(Icons.image, color: Colors.grey.shade400, size: 24),
      );
    }

    return GestureDetector(
      onTap: () {
        // 이미지 상세 화면으로 이동
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => ImageDetailScreen(
              imagePath: _lastCapturedImagePath!,
              imageTitle: _lastCapturedImagePath!.split('/').last,
            ),
          ),
        );
      },
      child: Stack(
        children: [
          Container(
            width: 56,
            height: 56,
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(16),
              image: DecorationImage(
                image: FileImage(File(_lastCapturedImagePath!)),
                fit: BoxFit.cover,
              ),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.1),
                  blurRadius: 8,
                  offset: const Offset(0, 2),
                ),
              ],
            ),
          ),
          // 파일 개수 표시
          if (_lastCapturedFolderFileCount > 0)
            Positioned(
              top: 4,
              right: 4,
              child: Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 6,
                  vertical: 3,
                ),
                decoration: BoxDecoration(
                  gradient: AppTheme.primaryGradient,
                  borderRadius: BorderRadius.circular(8),
                  boxShadow: [
                    BoxShadow(
                      color: AppTheme.primaryColor.withOpacity(0.3),
                      blurRadius: 4,
                      offset: const Offset(0, 2),
                    ),
                  ],
                ),
                child: Text(
                  '$_lastCapturedFolderFileCount',
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 10,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }

  /// 플래시 토글
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

  /// 이미지 캡처
  /// 고해상도로 촬영하고 4각형 영역을 잘라내어 저장
  Future<void> _captureImage() async {
    final controller = _cameraController;
    final result = _latestResult;

    if (controller == null || !controller.value.isInitialized) {
      return;
    }

    if (result == null ||
        result.corners == null ||
        result.corners!.length < 4) {
      _showSnackBar('문서가 제대로 인식되지 않았습니다.');
      return;
    }

    try {
      // 1. 이미지 스트림 중지
      await _stopImageStream();

      // 2. 고해상도 사진 촬영
      final xfile = await controller.takePicture();
      final imageBytes = await xfile.readAsBytes();
      final capturedImage = img.decodeImage(imageBytes);

      if (capturedImage == null) {
        throw StateError('Failed to decode captured image');
      }

      // 3. 방향 적용
      final oriented = _applyOrientation(capturedImage);

      // 4. 4각형 영역 잘라내기
      final croppedImage = _cropImageWithPerspective(oriented, result.corners!);

      // 5. 선택된 폴더에 저장
      final savedPath = await _saveImage(croppedImage, _selectedFolder);

      // 6. 폴더 파일 개수 업데이트
      await _loadFolders();

      setState(() {
        _lastCapturedImagePath = savedPath;
        _lastCapturedFolderFileCount = _folderFileCounts[_selectedFolder] ?? 0;
      });

      _showSnackBar('이미지가 저장되었습니다.');

      // 6. 이미지 스트림 재개
      await _startImageStream();
    } catch (e) {
      _showSnackBar('이미지 캡처 실패: $e');
      await _startImageStream();
    }
  }

  /// 원근 변환을 사용하여 이미지 크롭
  img.Image _cropImageWithPerspective(img.Image source, List<Corner> corners) {
    if (corners.length != 4) {
      // 4개 미만의 코너인 경우 바운딩 박스로 크롭
      final minX = corners
          .map((c) => c.x)
          .reduce((a, b) => a < b ? a : b)
          .toInt();
      final minY = corners
          .map((c) => c.y)
          .reduce((a, b) => a < b ? a : b)
          .toInt();
      final maxX = corners
          .map((c) => c.x)
          .reduce((a, b) => a > b ? a : b)
          .toInt();
      final maxY = corners
          .map((c) => c.y)
          .reduce((a, b) => a > b ? a : b)
          .toInt();

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

    // 4개의 코너로 원근 변환 사용
    final warped = PerspectiveTransform.warpPerspective(source, corners);
    return warped ?? source;
  }

  /// 이미지를 폴더에 저장
  /// 순차적인 숫자 파일명으로 PNG 저장
  Future<String> _saveImage(img.Image image, String folderName) async {
    final directory = await getApplicationDocumentsDirectory();
    final folderPath = '${directory.path}/ml_scanner/$folderName';
    final folder = Directory(folderPath);

    if (!await folder.exists()) {
      await folder.create(recursive: true);
    }

    // 폴더 내 파일 개수 확인하여 순차 번호 결정
    final files = await folder.list().toList();
    final imageFiles = files.whereType<File>().where((file) {
      return file.path.endsWith('.png');
    }).toList();

    final fileNumber = imageFiles.length + 1;
    final fileName = '$fileNumber.png';
    final filePath = '$folderPath/$fileName';

    final pngBytes = img.encodePng(image);
    final file = File(filePath);
    await file.writeAsBytes(pngBytes);

    return filePath;
  }

  /// 스낵바 표시
  void _showSnackBar(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), duration: const Duration(seconds: 2)),
    );
  }

  /// 폴더 목록 로드
  Future<void> _loadFolders() async {
    final directory = await getApplicationDocumentsDirectory();
    final basePath = '${directory.path}/ml_scanner';
    final baseDir = Directory(basePath);

    if (!await baseDir.exists()) {
      await baseDir.create(recursive: true);
      // 기본 폴더 생성
      final defaultFolder = Directory('$basePath/기본');
      await defaultFolder.create();
      return;
    }

    final entities = await baseDir.list().toList();
    final folderData = <String, int>{};

    for (final entity in entities.whereType<Directory>()) {
      final folderName = entity.path.split('/').last;
      final files = await entity.list().toList();
      final fileCount = files.whereType<File>().where((file) {
        return file.path.endsWith('.png') || file.path.endsWith('.pdf');
      }).length;
      folderData[folderName] = fileCount;
    }

    if (folderData.isEmpty) {
      final defaultFolder = Directory('$basePath/기본');
      await defaultFolder.create();
      folderData['기본'] = 0;
    }

    setState(() {
      _folderFileCounts.clear();
      _folderFileCounts.addAll(folderData);
      if (!_folderFileCounts.containsKey(_selectedFolder)) {
        _selectedFolder = _folderFileCounts.keys.first;
      }
    });
  }

  /// 코너 애니메이션 업데이트
  /// 확신도가 충분히 높을 때만 애니메이션 시작
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

  /// 애니메이션된 코너 좌표 가져오기
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

  /// 이미지에 센서 방향 적용
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

/// 에러 상태 위젯
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

/// 외곽선 페인터
/// 확신도에 따라 빨간색/초록색 윤곽선 표시
class _OutlinePainter extends CustomPainter {
  const _OutlinePainter({
    this.polygon,
    this.corners,
    required this.maskWidth,
    required this.maskHeight,
    required this.confidence,
  });

  final List<Corner>? polygon;
  final List<Corner>? corners;
  final double maskWidth;
  final double maskHeight;
  final double confidence;

  @override
  void paint(Canvas canvas, Size size) {
    // 4점 사각형 그리기
    if (corners != null && corners!.length == 4) {
      // 확신도 85% 이상이면 초록색, 미만이면 빨간색
      final isHighConfidence = confidence >= 0.85;
      final cornersPaint = Paint()
        ..color = isHighConfidence ? Colors.greenAccent : Colors.redAccent
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2; // 선 두께 50% 감소 (4 -> 2)

      final cornersPath = Path();
      for (var i = 0; i < corners!.length; i++) {
        final point = corners![i];
        final dx = (point.x / maskWidth) * size.width;
        final dy = (point.y / maskHeight) * size.height;
        if (i == 0) {
          cornersPath.moveTo(dx, dy);
        } else {
          cornersPath.lineTo(dx, dy);
        }
      }
      cornersPath.close();
      canvas.drawPath(cornersPath, cornersPaint);

      // 코너 포인트 그리기
      final pointPaint = Paint()
        ..color = isHighConfidence ? Colors.greenAccent : Colors.redAccent
        ..style = PaintingStyle.fill;

      for (final corner in corners!) {
        final dx = (corner.x / maskWidth) * size.width;
        final dy = (corner.y / maskHeight) * size.height;
        canvas.drawCircle(Offset(dx, dy), 4, pointPaint); // 크기도 약간 감소
      }
    }
  }

  @override
  bool shouldRepaint(covariant _OutlinePainter oldDelegate) {
    return oldDelegate.polygon != polygon ||
        oldDelegate.corners != corners ||
        oldDelegate.maskWidth != maskWidth ||
        oldDelegate.maskHeight != maskHeight ||
        oldDelegate.confidence != confidence;
  }
}

/// YUV420 이미지를 RGB 이미지로 변환
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
