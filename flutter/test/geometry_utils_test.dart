import 'package:flutter_test/flutter_test.dart';
import 'package:ml_scanner_app/utils/geometry_utils.dart';
import 'dart:math' as math;

void main() {
  group('GeometryUtils', () {
    test('approximateToQuadrilateral finds a square', () {
      final square = [
        const IntPoint(0, 0),
        const IntPoint(100, 0),
        const IntPoint(100, 100),
        const IntPoint(0, 100),
      ];
      
      // Add some noise points on the edges
      final points = [
        ...square,
        const IntPoint(50, 0),
        const IntPoint(100, 50),
        const IntPoint(50, 100),
        const IntPoint(0, 50),
      ];

      final result = GeometryUtils.approximateToQuadrilateral(points);
      expect(result, isNotNull);
      expect(result!.length, 4);
      
      // Check if corners match the square corners (order might vary but set should be same)
      final resultSet = result.map((p) => '${p.x},${p.y}').toSet();
      final expectedSet = square.map((p) => '${p.x},${p.y}').toSet();
      expect(resultSet, expectedSet);
    });

    test('orderCorners orders correctly', () {
      final corners = [
        const IntPoint(100, 100), // BR
        const IntPoint(0, 0),     // TL
        const IntPoint(0, 100),   // BL
        const IntPoint(100, 0),   // TR
      ];

      final ordered = GeometryUtils.orderCorners(corners);
      
      expect(ordered[0], const IntPoint(0, 0));     // TL
      expect(ordered[1], const IntPoint(100, 0));   // TR
      expect(ordered[2], const IntPoint(100, 100)); // BR
      expect(ordered[3], const IntPoint(0, 100));   // BL
    });

    test('checkParallelQuadrilateral detects parallel shapes', () {
      final square = [
        const IntPoint(0, 0),
        const IntPoint(100, 0),
        const IntPoint(100, 100),
        const IntPoint(0, 100),
      ];
      expect(GeometryUtils.checkParallelQuadrilateral(square), isTrue);

      final trapezoid = [
        const IntPoint(20, 0),
        const IntPoint(80, 0),
        const IntPoint(100, 100),
        const IntPoint(0, 100),
      ];
      // Trapezoid has one pair parallel, but function checks both pairs?
      // The implementation checks if BOTH pairs are parallel (parallelogram).
      // Wait, let's check the implementation logic.
      // It checks side1 // side3 AND side2 // side4.
      // So a trapezoid should be false.
      expect(GeometryUtils.checkParallelQuadrilateral(trapezoid), isFalse);
      
      final parallelogram = [
        const IntPoint(20, 0),
        const IntPoint(120, 0),
        const IntPoint(100, 100),
        const IntPoint(0, 100),
      ];
      expect(GeometryUtils.checkParallelQuadrilateral(parallelogram), isTrue);
    });
    
    test('approximateToQuadrilateral handles rotated square', () {
      // Rotated 45 degrees
      final center = const IntPoint(100, 100);
      final size = 50;
      final points = <IntPoint>[];
      
      for (var i = 0; i < 4; i++) {
        final angle = math.pi / 4 + (i * math.pi / 2);
        final x = (center.x + size * math.cos(angle)).round();
        final y = (center.y + size * math.sin(angle)).round();
        points.add(IntPoint(x, y));
      }
      
      final result = GeometryUtils.approximateToQuadrilateral(points);
      expect(result, isNotNull);
      expect(result!.length, 4);
      
      // Verify it found the 4 points
      final resultSet = result.map((p) => '${p.x},${p.y}').toSet();
      final expectedSet = points.map((p) => '${p.x},${p.y}').toSet();
      expect(resultSet, expectedSet);
    });
  });
}
