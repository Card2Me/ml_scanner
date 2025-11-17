import 'dart:io';

import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';

import '../theme/app_theme.dart';
import '../utils/file_utils.dart';
import 'gallery_screen.dart';

/// 폴더 관리자 화면
/// 폴더 목록 표시, 폴더 추가/삭제, 폴더별 파일 개수 표시
class FolderManagerScreen extends StatefulWidget {
  const FolderManagerScreen({super.key});

  @override
  State<FolderManagerScreen> createState() => _FolderManagerScreenState();
}

class _FolderManagerScreenState extends State<FolderManagerScreen> {
  List<FolderInfo> _folders = [];
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadFolders();
  }

  /// 폴더 목록 로드
  Future<void> _loadFolders() async {
    setState(() {
      _isLoading = true;
    });

    try {
      final directory = await getApplicationDocumentsDirectory();
      final basePath = '${directory.path}/ml_scanner';
      final baseDir = Directory(basePath);

      if (!await baseDir.exists()) {
        await baseDir.create(recursive: true);
      }

      final entities = await baseDir.list().toList();
      final folders = <FolderInfo>[];

      for (final entity in entities.whereType<Directory>()) {
        final folderName = entity.path.split('/').last;
        final files = await entity.list().toList();
        final imageCount = files.whereType<File>().where((file) {
          return file.path.endsWith('.png') || file.path.endsWith('.pdf');
        }).length;

        folders.add(FolderInfo(
          name: folderName,
          path: entity.path,
          fileCount: imageCount,
        ));
      }

      // 폴더가 없으면 기본 폴더 생성
      if (folders.isEmpty) {
        final defaultFolder = Directory('$basePath/기본');
        await defaultFolder.create();
        folders.add(FolderInfo(
          name: '기본',
          path: defaultFolder.path,
          fileCount: 0,
        ));
      }

      setState(() {
        _folders = folders;
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _isLoading = false;
      });
      _showSnackBar('폴더 로드 실패: $e');
    }
  }

  /// 새 폴더 추가
  Future<void> _addFolder() async {
    final controller = TextEditingController();
    final result = await showDialog<String>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('새 폴더'),
        content: TextField(
          controller: controller,
          decoration: const InputDecoration(
            labelText: '폴더 이름',
            hintText: '폴더 이름을 입력하세요',
          ),
          autofocus: true,
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('취소'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, controller.text),
            child: const Text('생성'),
          ),
        ],
      ),
    );

    if (result == null || result.trim().isEmpty) {
      return;
    }

    try {
      final directory = await getApplicationDocumentsDirectory();
      final basePath = '${directory.path}/ml_scanner';
      final newFolder = Directory('$basePath/$result');

      if (await newFolder.exists()) {
        _showSnackBar('이미 존재하는 폴더입니다');
        return;
      }

      await newFolder.create(recursive: true);
      _showSnackBar('폴더가 생성되었습니다');
      await _loadFolders();
    } catch (e) {
      _showSnackBar('폴더 생성 실패: $e');
    }
  }

  /// 폴더 삭제
  Future<void> _deleteFolder(FolderInfo folder) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('폴더 삭제'),
        content: Text('${folder.name} 폴더와 모든 파일을 삭제하시겠습니까?'),
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

    if (confirmed != true) {
      return;
    }

    try {
      final dir = Directory(folder.path);
      await dir.delete(recursive: true);
      _showSnackBar('폴더가 삭제되었습니다');
      await _loadFolders();
    } catch (e) {
      _showSnackBar('폴더 삭제 실패: $e');
    }
  }

  /// 폴더 메뉴 표시
  void _showFolderMenu(FolderInfo folder) {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (context) => Container(
        decoration: const BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.only(
            topLeft: Radius.circular(24),
            topRight: Radius.circular(24),
          ),
        ),
        child: SafeArea(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                margin: const EdgeInsets.only(top: 12, bottom: 8),
                width: 40,
                height: 4,
                decoration: BoxDecoration(
                  color: Colors.grey.shade300,
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                child: Text(
                  folder.name,
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: AppTheme.textPrimary,
                  ),
                ),
              ),
              const Divider(),
              _buildMenuTile(
                icon: Icons.folder_zip_rounded,
                title: 'ZIP으로 압축',
                gradient: AppTheme.primaryGradient,
                onTap: () {
                  Navigator.pop(context);
                  _zipFolder(folder);
                },
              ),
              _buildMenuTile(
                icon: Icons.share_rounded,
                title: '공유하기',
                gradient: AppTheme.secondaryGradient,
                onTap: () {
                  Navigator.pop(context);
                  _shareFolder(folder);
                },
              ),
              _buildMenuTile(
                icon: Icons.delete_rounded,
                title: '폴더 삭제',
                gradient: AppTheme.accentGradient,
                onTap: () {
                  Navigator.pop(context);
                  _deleteFolder(folder);
                },
              ),
              const SizedBox(height: 16),
            ],
          ),
        ),
      ),
    );
  }

  /// 메뉴 타일 빌드
  Widget _buildMenuTile({
    required IconData icon,
    required String title,
    required Gradient gradient,
    required VoidCallback onTap,
  }) {
    return ListTile(
      contentPadding: const EdgeInsets.symmetric(horizontal: 24, vertical: 4),
      leading: Container(
        width: 48,
        height: 48,
        decoration: BoxDecoration(
          gradient: gradient,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Icon(icon, color: Colors.white, size: 24),
      ),
      title: Text(
        title,
        style: const TextStyle(
          fontSize: 16,
          fontWeight: FontWeight.w500,
          color: AppTheme.textPrimary,
        ),
      ),
      trailing: Icon(
        Icons.arrow_forward_ios_rounded,
        size: 16,
        color: Colors.grey.shade400,
      ),
      onTap: onTap,
    );
  }

  /// 폴더를 ZIP으로 압축
  Future<void> _zipFolder(FolderInfo folder) async {
    if (folder.fileCount == 0) {
      _showSnackBar('폴더가 비어있습니다');
      return;
    }

    try {
      _showSnackBar('ZIP 압축 중...');

      final tempZipPath = await FileUtils.getTempFilePath('${folder.name}.zip');

      await FileUtils.zipFolder(
        folderPath: folder.path,
        outputPath: tempZipPath,
      );

      // ZIP 파일을 폴더에 복사
      final zipFile = File(tempZipPath);
      final targetPath = '${folder.path}/${folder.name}.zip';
      await zipFile.copy(targetPath);
      await zipFile.delete();

      _showSnackBar('ZIP 압축 완료');
      await _loadFolders();
    } catch (e) {
      _showSnackBar('ZIP 압축 실패: $e');
    }
  }

  /// 폴더 공유
  Future<void> _shareFolder(FolderInfo folder) async {
    if (folder.fileCount == 0) {
      _showSnackBar('폴더가 비어있습니다');
      return;
    }

    try {
      _showSnackBar('공유 준비 중...');

      // 폴더를 먼저 ZIP으로 압축
      final tempZipPath = await FileUtils.getTempFilePath('${folder.name}.zip');

      await FileUtils.zipFolder(
        folderPath: folder.path,
        outputPath: tempZipPath,
      );

      // ZIP 파일 공유
      await FileUtils.shareFiles(
        filePaths: [tempZipPath],
        subject: '${folder.name} 폴더',
      );

      // 임시 파일 삭제
      final zipFile = File(tempZipPath);
      if (await zipFile.exists()) {
        await zipFile.delete();
      }

      _showSnackBar('공유 완료');
    } catch (e) {
      _showSnackBar('공유 실패: $e');
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
        title: const Text(
          '폴더 관리',
          style: TextStyle(
            fontWeight: FontWeight.bold,
            fontSize: 20,
          ),
        ),
        backgroundColor: Colors.white,
        elevation: 0,
        actions: [
          Container(
            margin: const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
            decoration: BoxDecoration(
              gradient: AppTheme.primaryGradient,
              borderRadius: BorderRadius.circular(12),
            ),
            child: IconButton(
              icon: const Icon(Icons.add, color: Colors.white),
              onPressed: _addFolder,
              tooltip: '새 폴더',
            ),
          ),
        ],
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _folders.isEmpty
              ? const Center(
                  child: Text('폴더가 없습니다'),
                )
              : ListView.builder(
                  padding: const EdgeInsets.all(16),
                  itemCount: _folders.length,
                  itemBuilder: (context, index) {
                    final folder = _folders[index];
                    return Container(
                      margin: const EdgeInsets.only(bottom: 12),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(16),
                        boxShadow: AppTheme.cardShadow,
                      ),
                      child: ListTile(
                        contentPadding: const EdgeInsets.symmetric(
                          horizontal: 20,
                          vertical: 12,
                        ),
                        leading: Container(
                          width: 56,
                          height: 56,
                          decoration: BoxDecoration(
                            gradient: AppTheme.primaryGradient,
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: const Icon(
                            Icons.folder_rounded,
                            color: Colors.white,
                            size: 28,
                          ),
                        ),
                        title: Text(
                          folder.name,
                          style: const TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                            color: AppTheme.textPrimary,
                          ),
                        ),
                        subtitle: Padding(
                          padding: const EdgeInsets.only(top: 4),
                          child: Text(
                            '${folder.fileCount}개 파일',
                            style: TextStyle(
                              fontSize: 14,
                              color: AppTheme.textSecondary,
                            ),
                          ),
                        ),
                        trailing: Container(
                          decoration: BoxDecoration(
                            color: AppTheme.primaryColor.withOpacity(0.1),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: IconButton(
                            icon: Icon(
                              Icons.more_vert,
                              color: AppTheme.primaryColor,
                            ),
                            onPressed: () => _showFolderMenu(folder),
                          ),
                        ),
                        onTap: () {
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (context) => GalleryScreen(
                                folderName: folder.name,
                                folderPath: folder.path,
                              ),
                            ),
                          );
                        },
                      ),
                    );
                  },
                ),
    );
  }
}

/// 폴더 정보 클래스
class FolderInfo {
  const FolderInfo({
    required this.name,
    required this.path,
    required this.fileCount,
  });

  final String name;
  final String path;
  final int fileCount;
}
