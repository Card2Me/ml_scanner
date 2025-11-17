import 'dart:io';

import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';

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
      appBar: AppBar(
        title: Text(widget.folderName),
        actions: [
          if (_isSelectionMode) ...[
            IconButton(
              icon: const Icon(Icons.share),
              onPressed: _shareSelectedFiles,
              tooltip: '공유',
            ),
            IconButton(
              icon: const Icon(Icons.picture_as_pdf),
              onPressed: _convertToPdf,
              tooltip: 'PDF 변환',
            ),
            IconButton(
              icon: const Icon(Icons.delete),
              onPressed: _deleteSelectedFiles,
              tooltip: '삭제',
            ),
          ],
          IconButton(
            icon: Icon(_isSelectionMode ? Icons.close : Icons.check_box),
            onPressed: _toggleSelectionMode,
            tooltip: _isSelectionMode ? '선택 취소' : '선택',
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
                              borderRadius: BorderRadius.circular(8),
                              border: Border.all(
                                color: isSelected
                                    ? Colors.teal
                                    : Colors.grey.shade300,
                                width: isSelected ? 3 : 1,
                              ),
                            ),
                            child: ClipRRectangle(
                              borderRadius: BorderRadius.circular(8),
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
                              top: 4,
                              right: 4,
                              child: Container(
                                decoration: BoxDecoration(
                                  color: Colors.white,
                                  shape: BoxShape.circle,
                                  border: Border.all(color: Colors.grey),
                                ),
                                child: Icon(
                                  isSelected
                                      ? Icons.check_circle
                                      : Icons.circle_outlined,
                                  color: isSelected ? Colors.teal : Colors.grey,
                                  size: 24,
                                ),
                              ),
                            ),
                          // PDF 아이콘
                          if (file.isPdf)
                            Positioned(
                              bottom: 4,
                              left: 4,
                              child: Container(
                                padding: const EdgeInsets.all(4),
                                decoration: BoxDecoration(
                                  color: Colors.red,
                                  borderRadius: BorderRadius.circular(4),
                                ),
                                child: const Text(
                                  'PDF',
                                  style: TextStyle(
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
                  },
                ),
    );
  }

  /// PDF 썸네일 빌드
  Widget _buildPdfThumbnail(FileInfo file) {
    return Container(
      color: Colors.red.shade50,
      child: const Center(
        child: Icon(
          Icons.picture_as_pdf,
          size: 48,
          color: Colors.red,
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
