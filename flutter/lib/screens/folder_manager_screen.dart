import 'dart:io';

import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';

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
      builder: (context) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            ListTile(
              leading: const Icon(Icons.folder_zip),
              title: const Text('ZIP으로 압축'),
              onTap: () {
                Navigator.pop(context);
                _zipFolder(folder);
              },
            ),
            ListTile(
              leading: const Icon(Icons.share),
              title: const Text('공유하기'),
              onTap: () {
                Navigator.pop(context);
                _shareFolder(folder);
              },
            ),
            ListTile(
              leading: const Icon(Icons.delete, color: Colors.red),
              title: const Text('폴더 삭제', style: TextStyle(color: Colors.red)),
              onTap: () {
                Navigator.pop(context);
                _deleteFolder(folder);
              },
            ),
          ],
        ),
      ),
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
      appBar: AppBar(
        title: const Text('폴더 관리'),
        actions: [
          IconButton(
            icon: const Icon(Icons.add),
            onPressed: _addFolder,
            tooltip: '새 폴더',
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
                  itemCount: _folders.length,
                  itemBuilder: (context, index) {
                    final folder = _folders[index];
                    return Card(
                      margin: const EdgeInsets.symmetric(
                        horizontal: 16,
                        vertical: 8,
                      ),
                      child: ListTile(
                        leading: const Icon(Icons.folder, size: 40),
                        title: Text(
                          folder.name,
                          style: const TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        subtitle: Text('${folder.fileCount}개 파일'),
                        trailing: IconButton(
                          icon: const Icon(Icons.more_vert),
                          onPressed: () => _showFolderMenu(folder),
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
