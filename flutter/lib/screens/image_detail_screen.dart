import 'dart:io';

import 'package:flutter/material.dart';
import 'package:photo_view/photo_view.dart';
import 'package:photo_view/photo_view_gallery.dart';

import '../utils/file_utils.dart';
import 'gallery_screen.dart';

/// 이미지 상세 화면
/// 이미지를 전체 화면으로 표시하고 줌인/아웃 및 스와이프 지원
class ImageDetailScreen extends StatefulWidget {
  const ImageDetailScreen({
    required this.initialFile,
    required this.folderPath,
    super.key,
  });

  final File initialFile;
  final String folderPath;

  @override
  State<ImageDetailScreen> createState() => _ImageDetailScreenState();
}

class _ImageDetailScreenState extends State<ImageDetailScreen> {
  late PageController _pageController;
  List<File> _files = [];
  bool _isLoading = true;
  int _currentIndex = 0;

  @override
  void initState() {
    super.initState();
    _loadFiles();
  }

  @override
  void dispose() {
    _pageController.dispose();
    super.dispose();
  }

  /// 폴더 내 파일 목록 로드
  Future<void> _loadFiles() async {
    try {
      final dir = Directory(widget.folderPath);
      if (!await dir.exists()) {
        setState(() {
          _files = [widget.initialFile];
          _isLoading = false;
        });
        return;
      }

      final entities = await dir.list().toList();
      final files = <File>[];

      for (final entity in entities.whereType<File>()) {
        final fileName = entity.path.split('/').last.toLowerCase();
        final isPdf = fileName.endsWith('.pdf');
        final isImage =
            fileName.endsWith('.png') ||
            fileName.endsWith('.jpg') ||
            fileName.endsWith('.jpeg');

        if (isPdf || isImage) {
          files.add(entity);
        }
      }

      // 파일명으로 정렬
      files.sort(
        (a, b) => a.path.split('/').last.compareTo(b.path.split('/').last),
      );

      final initialIndex = files.indexWhere(
        (f) => f.path == widget.initialFile.path,
      );

      setState(() {
        _files = files;
        _currentIndex = initialIndex != -1 ? initialIndex : 0;
        _isLoading = false;
        _pageController = PageController(initialPage: _currentIndex);
      });
    } catch (e) {
      debugPrint('Error loading files: $e');
      setState(() {
        _files = [widget.initialFile];
        _isLoading = false;
        _pageController = PageController();
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return const Scaffold(
        backgroundColor: Colors.black,
        body: Center(child: CircularProgressIndicator()),
      );
    }

    final currentFile = _files[_currentIndex];
    final title = currentFile.path.split('/').last;
    final isPdf = title.toLowerCase().endsWith('.pdf');

    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        foregroundColor: Colors.white,
        title: Text(title),
        actions: [
          // 정보 버튼
          IconButton(
            icon: const Icon(Icons.info_outline),
            onPressed: () => _showImageProperties(currentFile),
            tooltip: '정보',
          ),
          // 메뉴 버튼
          PopupMenuButton<String>(
            icon: const Icon(Icons.more_vert),
            onSelected: (value) {
              if (value == 'folder') {
                _navigateToFolder();
              }
            },
            itemBuilder: (context) => [
              const PopupMenuItem(
                value: 'folder',
                child: Row(
                  children: [
                    Icon(Icons.folder_open, color: Colors.black87),
                    SizedBox(width: 8),
                    Text('폴더로 이동'),
                  ],
                ),
              ),
            ],
          ),
        ],
      ),
      body: PhotoViewGallery.builder(
        scrollPhysics: const BouncingScrollPhysics(),
        builder: (BuildContext context, int index) {
          final file = _files[index];

          if (isPdf) {
            return PhotoViewGalleryPageOptions.customChild(
              child: _buildPdfView(file.path.split('/').last),
              initialScale: PhotoViewComputedScale.contained,
              minScale: PhotoViewComputedScale.contained,
              maxScale: PhotoViewComputedScale.covered * 4.0,
              heroAttributes: PhotoViewHeroAttributes(tag: file.path),
            );
          }

          return PhotoViewGalleryPageOptions(
            imageProvider: FileImage(file),
            initialScale: PhotoViewComputedScale.contained,
            minScale: PhotoViewComputedScale.contained,
            maxScale: PhotoViewComputedScale.covered * 4.0,
            heroAttributes: PhotoViewHeroAttributes(tag: file.path),
          );
        },
        itemCount: _files.length,
        loadingBuilder: (context, event) =>
            const Center(child: CircularProgressIndicator()),
        backgroundDecoration: const BoxDecoration(color: Colors.black),
        pageController: _pageController,
        onPageChanged: (index) {
          setState(() {
            _currentIndex = index;
          });
        },
      ),
      bottomNavigationBar: BottomAppBar(
        color: Colors.black,
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            IconButton(
              icon: const Icon(Icons.share, color: Colors.white),
              onPressed: () => _shareFile(currentFile),
              tooltip: '공유',
            ),
            IconButton(
              icon: const Icon(Icons.delete, color: Colors.white),
              onPressed: () => _deleteImage(context),
              tooltip: '삭제',
            ),
          ],
        ),
      ),
    );
  }

  /// PDF 뷰 빌드
  Widget _buildPdfView(String title) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(Icons.picture_as_pdf, size: 100, color: Colors.red),
          const SizedBox(height: 16),
          Text(
            title,
            style: const TextStyle(color: Colors.white, fontSize: 18),
          ),
          const SizedBox(height: 8),
          const Text(
            'PDF 미리보기는 지원되지 않습니다',
            style: TextStyle(color: Colors.white70, fontSize: 14),
          ),
        ],
      ),
    );
  }

  /// 이미지 속성 표시
  Future<void> _showImageProperties(File file) async {
    if (!mounted) return;

    await showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(file.path.split('/').last),
        content: FutureBuilder<Map<String, String>>(
          future: FileUtils.getFileProperties(file),
          builder: (context, snapshot) {
            if (snapshot.connectionState != ConnectionState.done) {
              return const SizedBox(
                height: 100,
                child: Center(child: CircularProgressIndicator()),
              );
            }

            if (snapshot.hasError) {
              return Text('속성을 불러오는데 실패했습니다: ${snapshot.error}');
            }

            final props = snapshot.data!;
            return Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: props.entries.map((e) {
                return Padding(
                  padding: const EdgeInsets.symmetric(vertical: 4),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      SizedBox(
                        width: 80,
                        child: Text(
                          e.key,
                          style: const TextStyle(
                            fontWeight: FontWeight.bold,
                            color: Colors.grey,
                          ),
                        ),
                      ),
                      Expanded(
                        child: Text(
                          e.value,
                          style: const TextStyle(fontWeight: FontWeight.w500),
                        ),
                      ),
                    ],
                  ),
                );
              }).toList(),
            );
          },
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('확인'),
          ),
        ],
      ),
    );
  }

  /// 폴더로 이동
  void _navigateToFolder() {
    final folderName = widget.folderPath.split('/').last;
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(
        builder: (context) => GalleryScreen(
          folderName: folderName,
          folderPath: widget.folderPath,
        ),
      ),
    );
  }

  /// 파일 공유
  Future<void> _shareFile(File file) async {
    try {
      await FileUtils.shareFiles(filePaths: [file.path]);
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('공유 실패: $e')));
      }
    }
  }

  /// 이미지 삭제
  Future<void> _deleteImage(BuildContext context) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('파일 삭제'),
        content: const Text('이 파일을 삭제하시겠습니까?'),
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
      final file = _files[_currentIndex];
      await file.delete();

      setState(() {
        _files.removeAt(_currentIndex);
        if (_files.isEmpty) {
          Navigator.pop(context); // 파일이 없으면 화면 닫기
        } else if (_currentIndex >= _files.length) {
          _currentIndex = _files.length - 1;
          _pageController.jumpToPage(_currentIndex);
        }
      });

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('파일이 삭제되었습니다'),
            duration: Duration(seconds: 2),
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('파일 삭제 실패: $e'),
            duration: const Duration(seconds: 2),
          ),
        );
      }
    }
  }
}
