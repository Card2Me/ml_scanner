import 'dart:io';

import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';

import '../theme/app_theme.dart';
import 'gallery_screen.dart';

/// 폴더 관리 화면
class FolderManagerScreen extends StatefulWidget {
  const FolderManagerScreen({super.key});

  @override
  State<FolderManagerScreen> createState() => _FolderManagerScreenState();
}

class _FolderManagerScreenState extends State<FolderManagerScreen> {
  List<String> _folders = [];
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
      final appDir = Directory('${directory.path}/ml_scanner');

      if (!await appDir.exists()) {
        await appDir.create(recursive: true);
      }

      final entities = await appDir.list().toList();
      final folders = <String>[];

      for (final entity in entities.whereType<Directory>()) {
        final folderName = entity.path.split('/').last;
        folders.add(folderName);
      }

      // 기본 폴더가 없으면 생성
      if (!folders.contains('기본')) {
        await Directory('${appDir.path}/기본').create();
        folders.add('기본');
      }

      folders.sort();

      setState(() {
        _folders = folders;
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _isLoading = false;
      });
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('폴더 로드 실패: $e')));
      }
    }
  }

  /// 새 폴더 추가 다이얼로그 표시
  Future<void> _showAddFolderDialog() async {
    final controller = TextEditingController();

    await showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('새 폴더 추가'),
        content: TextField(
          controller: controller,
          decoration: const InputDecoration(
            labelText: '폴더 이름',
            hintText: '새 폴더 이름을 입력하세요',
          ),
          autofocus: true,
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('취소'),
          ),
          TextButton(
            onPressed: () async {
              final folderName = controller.text.trim();
              if (folderName.isEmpty) return;

              if (_folders.contains(folderName)) {
                ScaffoldMessenger.of(
                  context,
                ).showSnackBar(const SnackBar(content: Text('이미 존재하는 폴더입니다.')));
                return;
              }

              try {
                final directory = await getApplicationDocumentsDirectory();
                await Directory(
                  '${directory.path}/ml_scanner/$folderName',
                ).create(recursive: true);

                if (context.mounted) {
                  Navigator.pop(context);
                  _loadFolders();
                }
              } catch (e) {
                if (context.mounted) {
                  ScaffoldMessenger.of(
                    context,
                  ).showSnackBar(SnackBar(content: Text('폴더 생성 실패: $e')));
                }
              }
            },
            child: const Text('추가'),
          ),
        ],
      ),
    );
  }

  /// 폴더 삭제
  Future<void> _deleteFolder(String folderName) async {
    if (folderName == '기본') {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('기본 폴더는 삭제할 수 없습니다.')));
      return;
    }

    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('폴더 삭제'),
        content: Text("'$folderName' 폴더와 내부 파일이 모두 삭제됩니다.\n계속하시겠습니까?"),
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
      final directory = await getApplicationDocumentsDirectory();
      final folder = Directory('${directory.path}/ml_scanner/$folderName');

      if (await folder.exists()) {
        await folder.delete(recursive: true);
      }

      await _loadFolders();

      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(const SnackBar(content: Text('폴더가 삭제되었습니다.')));
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('폴더 삭제 실패: $e')));
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.backgroundColor,
      appBar: AppBar(
        title: const Text(
          '폴더 관리',
          style: TextStyle(fontWeight: FontWeight.bold, fontSize: 20),
        ),
        backgroundColor: Colors.white,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => Navigator.pop(context),
        ),
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : ListView.builder(
              padding: const EdgeInsets.all(16),
              itemCount: _folders.length,
              itemBuilder: (context, index) {
                final folderName = _folders[index];
                final isDefault = folderName == '기본';

                return Card(
                  margin: const EdgeInsets.only(bottom: 12),
                  elevation: 2,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: ListTile(
                    leading: Container(
                      padding: const EdgeInsets.all(8),
                      decoration: BoxDecoration(
                        color: AppTheme.primaryColor.withOpacity(0.1),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Icon(Icons.folder, color: AppTheme.primaryColor),
                    ),
                    title: Text(
                      folderName,
                      style: const TextStyle(
                        fontWeight: FontWeight.bold,
                        fontSize: 16,
                      ),
                    ),
                    trailing: isDefault
                        ? null
                        : IconButton(
                            icon: const Icon(
                              Icons.delete_outline,
                              color: Colors.red,
                            ),
                            onPressed: () => _deleteFolder(folderName),
                          ),
                    onTap: () async {
                      final directory =
                          await getApplicationDocumentsDirectory();
                      final folderPath =
                          '${directory.path}/ml_scanner/$folderName';

                      if (context.mounted) {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (context) => GalleryScreen(
                              folderName: folderName,
                              folderPath: folderPath,
                            ),
                          ),
                        );
                      }
                    },
                  ),
                );
              },
            ),
      floatingActionButton: FloatingActionButton(
        onPressed: _showAddFolderDialog,
        backgroundColor: AppTheme.primaryColor,
        child: const Icon(Icons.add, color: Colors.white),
      ),
    );
  }
}
