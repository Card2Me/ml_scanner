import 'dart:io';

import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';

import '../theme/app_theme.dart';
import '../utils/file_utils.dart';
import 'image_detail_screen.dart';

/// 갤러리 화면
/// 특정 폴더의 이미지/PDF 목록을 그리드로 표시
class GalleryScreen extends StatefulWidget {
  const GalleryScreen({
    required this.folderName,
    required this.folderPath,
    super.key,
  });

  final String folderName;
  final String folderPath;

  @override
  State<GalleryScreen> createState() => _GalleryScreenState();
}

class _GalleryScreenState extends State<GalleryScreen> {
  List<FileInfo> _files = [];
  bool _isLoading = true;
  bool _isSelectionMode = false;
  final Set<int> _selectedIndices = {};

  @override
  void initState() {
    super.initState();
    _loadFiles();
  }

  /// 파일 목록 로드
  Future<void> _loadFiles() async {
    setState(() {
      _isLoading = true;
    });

    try {
      final dir = Directory(widget.folderPath);
      if (!await dir.exists()) {
        setState(() {
          _files = [];
          _isLoading = false;
        });
        return;
      }

      final entities = await dir.list().toList();
      final files = <FileInfo>[];

      for (final entity in entities.whereType<File>()) {
        final fileName = entity.path.split('/').last;
        final isPdf = fileName.endsWith('.pdf');
        final isImage = fileName.endsWith('.png') ||
            fileName.endsWith('.jpg') ||
            fileName.endsWith('.jpeg');

        if (isPdf || isImage) {
          files.add(FileInfo(
            path: entity.path,
            name: fileName,
            isPdf: isPdf,
          ));
        }
      }

      // 파일명으로 정렬
      files.sort((a, b) => a.name.compareTo(b.name));

      setState(() {
        _files = files;
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _isLoading = false;
      });
      _showSnackBar('파일 로드 실패: $e');
    }
  }

  /// 선택 모드 토글
  void _toggleSelectionMode() {
    setState(() {
      _isSelectionMode = !_isSelectionMode;
      if (!_isSelectionMode) {
        _selectedIndices.clear();
      }
    });
  }

  /// 선택된 파일 삭제
  Future<void> _deleteSelectedFiles() async {
    if (_selectedIndices.isEmpty) return;

    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('파일 삭제'),
        content: Text('${_selectedIndices.length}개 파일을 삭제하시겠습니까?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('취소'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            style: TextButton.styleFrom(foregroundColor: Colors.red),
            child: const Text('삭제'),
          ),
        ],
      ),
    );

    if (confirmed != true) return;

    try {
      for (final index in _selectedIndices) {
        final file = File(_files[index].path);
        await file.delete();
      }

      _showSnackBar('${_selectedIndices.length}개 파일 삭제됨');
      _selectedIndices.clear();
      _isSelectionMode = false;
      await _loadFiles();
    } catch (e) {
      _showSnackBar('파일 삭제 실패: $e');
    }
  }

  /// 선택된 파일 공유
  Future<void> _shareSelectedFiles() async {
    if (_selectedIndices.isEmpty) return;

    try {
      final filePaths = _selectedIndices
          .map((index) => _files[index].path)
          .toList();

      await FileUtils.shareFiles(
        filePaths: filePaths,
        subject: '${widget.folderName} 파일',
      );

      _showSnackBar('파일 공유 완료');
    } catch (e) {
      _showSnackBar('파일 공유 실패: $e');
    }
  }

  /// 선택된 파일을 PDF로 변환
  Future<void> _convertToPdf() async {
    if (_selectedIndices.isEmpty) return;

    // PDF가 아닌 이미지 파일만 선택
    final imagePaths = _selectedIndices
        .map((index) => _files[index])
        .where((file) => !file.isPdf)
        .map((file) => file.path)
        .toList();

    if (imagePaths.isEmpty) {
      _showSnackBar('이미지 파일을 선택해주세요');
      return;
    }

    try {
      _showSnackBar('PDF 변환 중...');

      // PDF 파일명 생성
      final timestamp = DateTime.now().millisecondsSinceEpoch;
      final pdfFileName = 'scan_$timestamp.pdf';
      final pdfPath = '${widget.folderPath}/$pdfFileName';

      await FileUtils.convertImagesToPdf(
        imagePaths: imagePaths,
        outputPath: pdfPath,
      );

      _showSnackBar('PDF 변환 완료');
      _selectedIndices.clear();
      _isSelectionMode = false;
      await _loadFiles();
    } catch (e) {
      _showSnackBar('PDF 변환 실패: $e');
    }
  }

  void _showSnackBar(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), duration: const Duration(seconds: 2)),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.backgroundColor,
      appBar: AppBar(
        title: Text(
          widget.folderName,
          style: const TextStyle(
            fontWeight: FontWeight.bold,
            fontSize: 20,
          ),
        ),
        backgroundColor: Colors.white,
        elevation: 0,
        actions: [
          if (_isSelectionMode) ...[
            Container(
              margin: const EdgeInsets.symmetric(horizontal: 4, vertical: 8),
              decoration: BoxDecoration(
                gradient: AppTheme.secondaryGradient,
                borderRadius: BorderRadius.circular(12),
              ),
              child: IconButton(
                icon: const Icon(Icons.share, color: Colors.white),
                onPressed: _shareSelectedFiles,
                tooltip: '공유',
              ),
            ),
            Container(
              margin: const EdgeInsets.symmetric(horizontal: 4, vertical: 8),
              decoration: BoxDecoration(
                gradient: AppTheme.primaryGradient,
                borderRadius: BorderRadius.circular(12),
              ),
              child: IconButton(
                icon: const Icon(Icons.picture_as_pdf, color: Colors.white),
                onPressed: _convertToPdf,
                tooltip: 'PDF 변환',
              ),
            ),
            Container(
              margin: const EdgeInsets.symmetric(horizontal: 4, vertical: 8),
              decoration: BoxDecoration(
                gradient: AppTheme.accentGradient,
                borderRadius: BorderRadius.circular(12),
              ),
              child: IconButton(
                icon: const Icon(Icons.delete, color: Colors.white),
                onPressed: _deleteSelectedFiles,
                tooltip: '삭제',
              ),
            ),
          ],
          Container(
            margin: const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
            decoration: BoxDecoration(
              color: _isSelectionMode ? Colors.grey.shade200 : AppTheme.primaryColor.withOpacity(0.1),
              borderRadius: BorderRadius.circular(12),
            ),
            child: IconButton(
              icon: Icon(
                _isSelectionMode ? Icons.close : Icons.check_box_outlined,
                color: _isSelectionMode ? Colors.grey.shade700 : AppTheme.primaryColor,
              ),
              onPressed: _toggleSelectionMode,
              tooltip: _isSelectionMode ? '선택 취소' : '선택',
            ),
          ),
        ],
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _files.isEmpty
              ? const Center(
                  child: Text('파일이 없습니다'),
                )
              : GridView.builder(
                  padding: const EdgeInsets.all(8),
                  gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                    crossAxisCount: 3,
                    crossAxisSpacing: 8,
                    mainAxisSpacing: 8,
                  ),
                  itemCount: _files.length,
                  itemBuilder: (context, index) {
                    final file = _files[index];
                    final isSelected = _selectedIndices.contains(index);

                    return GestureDetector(
                      onTap: () {
                        if (_isSelectionMode) {
                          setState(() {
                            if (isSelected) {
                              _selectedIndices.remove(index);
                            } else {
                              _selectedIndices.add(index);
                            }
                          });
                        } else {
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (context) => ImageDetailScreen(
                                imagePath: file.path,
                                imageTitle: file.name,
                              ),
                            ),
                          );
                        }
                      },
                      child: Stack(
                        fit: StackFit.expand,
                        children: [
                          // 썸네일
                          Container(
                            decoration: BoxDecoration(
                              borderRadius: BorderRadius.circular(12),
                              border: Border.all(
                                color: isSelected
                                    ? AppTheme.primaryColor
                                    : Colors.transparent,
                                width: isSelected ? 3 : 0,
                              ),
                              boxShadow: isSelected
                                  ? [
                                      BoxShadow(
                                        color: AppTheme.primaryColor.withOpacity(0.3),
                                        blurRadius: 12,
                                        offset: const Offset(0, 4),
                                      ),
                                    ]
                                  : AppTheme.cardShadow,
                            ),
                            child: ClipRRectangle(
                              borderRadius: BorderRadius.circular(12),
                              child: file.isPdf
                                  ? _buildPdfThumbnail(file)
                                  : Image.file(
                                      File(file.path),
                                      fit: BoxFit.cover,
                                    ),
                            ),
                          ),
                          // 선택 체크박스
                          if (_isSelectionMode)
                            Positioned(
                              top: 8,
                              right: 8,
                              child: Container(
                                width: 28,
                                height: 28,
                                decoration: BoxDecoration(
                                  gradient: isSelected ? AppTheme.primaryGradient : null,
                                  color: isSelected ? null : Colors.white,
                                  shape: BoxShape.circle,
                                  border: Border.all(
                                    color: isSelected ? Colors.transparent : Colors.grey.shade400,
                                    width: 2,
                                  ),
                                  boxShadow: [
                                    BoxShadow(
                                      color: Colors.black.withOpacity(0.2),
                                      blurRadius: 4,
                                      offset: const Offset(0, 2),
                                    ),
                                  ],
                                ),
                                child: Icon(
                                  isSelected ? Icons.check : Icons.circle_outlined,
                                  color: isSelected ? Colors.white : Colors.grey.shade400,
                                  size: 16,
                                ),
                              ),
                            ),
                          // PDF 아이콘
                          if (file.isPdf)
                            Positioned(
                              bottom: 8,
                              left: 8,
                              child: Container(
                                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                                decoration: BoxDecoration(
                                  gradient: AppTheme.accentGradient,
                                  borderRadius: BorderRadius.circular(8),
                                  boxShadow: [
                                    BoxShadow(
                                      color: AppTheme.accentColor.withOpacity(0.3),
                                      blurRadius: 8,
                                      offset: const Offset(0, 2),
                                    ),
                                  ],
                                ),
                                child: const Text(
                                  'PDF',
                                  style: TextStyle(
                                    color: Colors.white,
                                    fontSize: 10,
                                    fontWeight: FontWeight.bold,
                                    letterSpacing: 0.5,
                                  ),
                                ),
                              ),
                            ),
                        ],
                      ),
                    );
                  },
                ),
    );
  }

  /// PDF 썸네일 빌드
  Widget _buildPdfThumbnail(FileInfo file) {
    return Container(
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [
            AppTheme.accentColor.withOpacity(0.1),
            AppTheme.accentColor.withOpacity(0.05),
          ],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
      ),
      child: Center(
        child: Icon(
          Icons.picture_as_pdf_rounded,
          size: 48,
          color: AppTheme.accentColor,
        ),
      ),
    );
  }
}

/// ClipRRect 대체 (오타 수정)
class ClipRRectangle extends StatelessWidget {
  const ClipRRectangle({
    required this.child,
    required this.borderRadius,
    super.key,
  });

  final Widget child;
  final BorderRadius borderRadius;

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: borderRadius,
      child: child,
    );
  }
}

/// 파일 정보 클래스
class FileInfo {
  const FileInfo({
    required this.path,
    required this.name,
    required this.isPdf,
  });

  final String path;
  final String name;
  final bool isPdf;
}
