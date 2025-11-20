import 'dart:io';

import 'package:archive/archive_io.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';
import 'package:pdf/pdf.dart';
import 'package:pdf/widgets.dart' as pw;
import 'package:share_plus/share_plus.dart';

/// 파일 관련 유틸리티 함수 모음
class FileUtils {
  /// 이미지 파일들을 PDF로 변환
  /// [imagePaths] 변환할 이미지 파일 경로 목록
  /// [outputPath] 생성할 PDF 파일 경로
  static Future<String> convertImagesToPdf({
    required List<String> imagePaths,
    required String outputPath,
  }) async {
    final pdf = pw.Document();

    for (final imagePath in imagePaths) {
      // 이미지 파일 읽기
      final imageFile = File(imagePath);
      if (!await imageFile.exists()) {
        continue;
      }

      final imageBytes = await imageFile.readAsBytes();
      final decodedImage = img.decodeImage(imageBytes);

      if (decodedImage == null) {
        continue;
      }

      // PNG로 인코딩 (PDF 라이브러리가 PNG를 지원)
      final pngBytes = img.encodePng(decodedImage);

      // PDF 이미지 생성
      final pdfImage = pw.MemoryImage(pngBytes);

      // PDF 페이지 추가
      pdf.addPage(
        pw.Page(
          pageFormat: PdfPageFormat.a4,
          build: (context) {
            return pw.Center(child: pw.Image(pdfImage, fit: pw.BoxFit.contain));
          },
        ),
      );
    }

    // PDF 파일 저장
    final file = File(outputPath);
    await file.writeAsBytes(await pdf.save());

    return outputPath;
  }

  /// 폴더를 ZIP으로 압축
  /// [folderPath] 압축할 폴더 경로
  /// [outputPath] 생성할 ZIP 파일 경로
  static Future<String> zipFolder({
    required String folderPath,
    required String outputPath,
  }) async {
    final dir = Directory(folderPath);
    if (!await dir.exists()) {
      throw Exception('폴더가 존재하지 않습니다');
    }

    // Archive 생성
    final archive = Archive();

    // 폴더 내 모든 파일 추가
    final entities = await dir.list(recursive: true).toList();
    for (final entity in entities) {
      if (entity is File) {
        final file = entity;
        final filename = file.path.split('/').last;
        final bytes = await file.readAsBytes();
        archive.addFile(ArchiveFile(filename, bytes.length, bytes));
      }
    }

    // ZIP 파일로 인코딩
    final zipEncoder = ZipEncoder();
    final zipBytes = zipEncoder.encode(archive);

    if (zipBytes == null) {
      throw Exception('ZIP 압축 실패');
    }

    // ZIP 파일 저장
    final zipFile = File(outputPath);
    await zipFile.writeAsBytes(zipBytes);

    return outputPath;
  }

  /// 선택된 파일들을 ZIP으로 압축
  /// [filePaths] 압축할 파일 경로 목록
  /// [outputPath] 생성할 ZIP 파일 경로
  static Future<String> zipFiles({
    required List<String> filePaths,
    required String outputPath,
  }) async {
    if (filePaths.isEmpty) {
      throw Exception('압축할 파일이 없습니다');
    }

    // Archive 생성
    final archive = Archive();

    for (final path in filePaths) {
      final file = File(path);
      if (await file.exists()) {
        final filename = file.path.split('/').last;
        final bytes = await file.readAsBytes();
        archive.addFile(ArchiveFile(filename, bytes.length, bytes));
      }
    }

    // ZIP 파일로 인코딩
    final zipEncoder = ZipEncoder();
    final zipBytes = zipEncoder.encode(archive);

    if (zipBytes == null) {
      throw Exception('ZIP 압축 실패');
    }

    // ZIP 파일 저장
    final zipFile = File(outputPath);
    await zipFile.writeAsBytes(zipBytes);

    return outputPath;
  }

  /// 파일 공유
  /// [filePaths] 공유할 파일 경로 목록
  /// [subject] 공유 제목 (선택)
  static Future<void> shareFiles({
    required List<String> filePaths,
    String? subject,
  }) async {
    if (filePaths.isEmpty) {
      throw Exception('공유할 파일이 없습니다');
    }

    final xFiles = filePaths.map((path) => XFile(path)).toList();

    await Share.shareXFiles(xFiles, subject: subject);
  }

  /// 임시 파일 경로 생성
  /// [fileName] 파일 이름
  static Future<String> getTempFilePath(String fileName) async {
    final tempDir = await getTemporaryDirectory();
    return '${tempDir.path}/$fileName';
  }

  /// 폴더 내 파일 개수 가져오기
  /// [folderPath] 폴더 경로
  static Future<int> getFileCount(String folderPath) async {
    final dir = Directory(folderPath);
    if (!await dir.exists()) {
      return 0;
    }

    final files = await dir.list().toList();
    return files.whereType<File>().where((file) {
      return file.path.endsWith('.png') ||
          file.path.endsWith('.jpg') ||
          file.path.endsWith('.jpeg') ||
          file.path.endsWith('.pdf');
    }).length;
  }

  /// 파일 크기를 읽기 쉬운 형식으로 변환
  /// [bytes] 바이트 크기
  static String formatFileSize(int bytes) {
    if (bytes < 1024) {
      return '$bytes B';
    } else if (bytes < 1024 * 1024) {
      return '${(bytes / 1024).toStringAsFixed(1)} KB';
    } else if (bytes < 1024 * 1024 * 1024) {
      return '${(bytes / (1024 * 1024)).toStringAsFixed(1)} MB';
    } else {
      return '${(bytes / (1024 * 1024 * 1024)).toStringAsFixed(1)} GB';
    }
  }

  /// 파일 속성 가져오기
  static Future<Map<String, String>> getFileProperties(File file) async {
    final stat = await file.stat();
    final size = formatFileSize(stat.size);
    final date = formatDate(stat.modified);
    final isPdf = file.path.toLowerCase().endsWith('.pdf');

    String resolution = 'N/A';
    if (!isPdf) {
      try {
        final bytes = await file.readAsBytes();
        final image = img.decodeImage(bytes);
        if (image != null) {
          resolution = '${image.width} x ${image.height}';
        }
      } catch (_) {
        resolution = 'Unknown';
      }
    }

    return {
      '파일 크기': size,
      '수정 날짜': date,
      if (!isPdf) '해상도': resolution,
      '경로': file.path,
    };
  }

  /// 날짜 포맷팅
  static String formatDate(DateTime date) {
    return '${date.year}-${date.month.toString().padLeft(2, '0')}-${date.day.toString().padLeft(2, '0')} '
        '${date.hour.toString().padLeft(2, '0')}:${date.minute.toString().padLeft(2, '0')}';
  }
}
