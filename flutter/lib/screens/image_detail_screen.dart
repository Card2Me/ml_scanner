import 'dart:io';

import 'package:flutter/material.dart';
import 'package:photo_view/photo_view.dart';

/// 이미지 상세 화면
/// 이미지를 전체 화면으로 표시하고 줌인/아웃 지원
class ImageDetailScreen extends StatelessWidget {
  const ImageDetailScreen({
    required this.imagePath,
    required this.imageTitle,
    super.key,
  });

  final String imagePath;
  final String imageTitle;

  @override
  Widget build(BuildContext context) {
    final isPdf = imagePath.endsWith('.pdf');

    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        foregroundColor: Colors.white,
        title: Text(imageTitle),
        actions: [
          IconButton(
            icon: const Icon(Icons.share),
            onPressed: () {
              // TODO: 공유 기능
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(
                  content: Text('공유 기능은 곧 추가됩니다'),
                  duration: Duration(seconds: 2),
                ),
              );
            },
            tooltip: '공유',
          ),
          IconButton(
            icon: const Icon(Icons.delete),
            onPressed: () => _deleteImage(context),
            tooltip: '삭제',
          ),
        ],
      ),
      body: isPdf
          ? _buildPdfView()
          : PhotoView(
              imageProvider: FileImage(File(imagePath)),
              minScale: PhotoViewComputedScale.contained,
              maxScale: PhotoViewComputedScale.covered * 4.0,
              initialScale: PhotoViewComputedScale.contained,
              backgroundDecoration: const BoxDecoration(
                color: Colors.black,
              ),
            ),
    );
  }

  /// PDF 뷰 빌드
  Widget _buildPdfView() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(
            Icons.picture_as_pdf,
            size: 100,
            color: Colors.red,
          ),
          const SizedBox(height: 16),
          Text(
            imageTitle,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 18,
            ),
          ),
          const SizedBox(height: 8),
          const Text(
            'PDF 미리보기는 지원되지 않습니다',
            style: TextStyle(
              color: Colors.white70,
              fontSize: 14,
            ),
          ),
        ],
      ),
    );
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
      final file = File(imagePath);
      await file.delete();

      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('파일이 삭제되었습니다'),
            duration: Duration(seconds: 2),
          ),
        );
        Navigator.pop(context);
      }
    } catch (e) {
      if (context.mounted) {
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
