import 'dart:math' as math;
import 'package:image/image.dart' as img;

/// 정수 좌표를 나타내는 클래스
class IntPoint {
  const IntPoint(this.x, this.y);

  final int x;
  final int y;

  @override
  String toString() => '($x, $y)';

  @override
  bool operator ==(Object other) =>
      other is IntPoint && other.x == x && other.y == y;

  @override
  int get hashCode => Object.hash(x, y);
}

/// 실수 좌표를 나타내는 코너 클래스
class Corner {
  const Corner(this.x, this.y);

  final double x;
  final double y;

  /// 좌표를 스케일 팩터로 확대/축소
  Corner scale(double scaleX, double scaleY) {
    return Corner(x * scaleX, y * scaleY);
  }

  @override
  String toString() => '(${x.toStringAsFixed(1)}, ${y.toStringAsFixed(1)})';
}

/// 기하학 연산 유틸리티
class GeometryUtils {
  /// Douglas-Peucker 알고리즘을 사용하여 다각형을 4점 사각형으로 근사
  static List<IntPoint>? approximateToQuadrilateral(List<IntPoint> polygon) {
    if (polygon.length < 3) {
      return null;
    }
    if (polygon.length == 4) {
      return orderCorners(polygon);
    }

    // 1. Convex Hull 구하기 (이미 Convex Hull이 들어온다고 가정하지만 안전을 위해)
    final hull = convexHull(polygon);

    if (hull.length < 4) {
      // 점이 4개 미만이면 null
      return null;
    }

    if (hull.length == 4) {
      return orderCorners(hull);
    }

    // 2. 점이 너무 많으면 Douglas-Peucker로 단순화
    var simplified = hull;
    if (simplified.length > 20) {
      simplified = _simplifyPolygon(hull, 2.0); // epsilon 2.0
    }

    // 3. 4개의 점을 선택하여 가장 큰 면적을 가지는 사각형 찾기
    // 점의 개수가 적으므로 모든 조합(또는 최적화된 방식)을 시도해볼 수 있음
    // 여기서는 단순화된 hull에서 가장 넓은 면적을 만드는 4점을 찾습니다.
    return _findLargestQuadrilateral(simplified);
  }

  /// 다각형 단순화 (Douglas-Peucker)
  static List<IntPoint> _simplifyPolygon(
    List<IntPoint> points,
    double epsilon,
  ) {
    if (points.length < 3) return points;

    // Find the point with maximum distance from line segment
    var maxDist = 0.0;
    var maxIndex = 0;
    final start = points.first;
    final end = points.last;

    for (var i = 1; i < points.length - 1; i++) {
      final dist = _perpendicularDistance(points[i], start, end);
      if (dist > maxDist) {
        maxDist = dist;
        maxIndex = i;
      }
    }

    if (maxDist > epsilon) {
      final left = _simplifyPolygon(points.sublist(0, maxIndex + 1), epsilon);
      final right = _simplifyPolygon(points.sublist(maxIndex), epsilon);
      return [...left.sublist(0, left.length - 1), ...right];
    } else {
      return [start, end];
    }
  }

  static double _perpendicularDistance(
    IntPoint point,
    IntPoint lineStart,
    IntPoint lineEnd,
  ) {
    final dx = lineEnd.x - lineStart.x;
    final dy = lineEnd.y - lineStart.y;
    final norm = math.sqrt(dx * dx + dy * dy);
    if (norm == 0) return 0;
    return ((point.y - lineStart.y) * dx - (point.x - lineStart.x) * dy).abs() /
        norm;
  }

  /// 가장 큰 면적을 가지는 4각형 찾기
  static List<IntPoint>? _findLargestQuadrilateral(List<IntPoint> hull) {
    if (hull.length < 4) return null;

    // 점이 많지 않다고 가정 (Douglas-Peucker 후) -> O(N^4)도 괜찮지만 N이 20이하면 160,000.
    // N이 크면 부담스러움. N을 10~15 정도로 줄이는 것이 좋음.

    // 만약 점이 여전히 많다면, 다시 단순화
    var candidates = hull;
    double epsilon = 2.0;
    while (candidates.length > 12) {
      candidates = _simplifyPolygon(candidates, epsilon);
      epsilon += 2.0;
      if (epsilon > 20) break; // 안전장치
    }

    if (candidates.length < 4) return null;
    if (candidates.length == 4) return orderCorners(candidates);

    double maxArea = 0;
    List<IntPoint>? bestQuad;

    final n = candidates.length;
    for (var i = 0; i < n; i++) {
      for (var j = i + 1; j < n; j++) {
        for (var k = j + 1; k < n; k++) {
          for (var l = k + 1; l < n; l++) {
            final p1 = candidates[i];
            final p2 = candidates[j];
            final p3 = candidates[k];
            final p4 = candidates[l];

            // 면적 계산 (두 삼각형의 합)
            final area = _triangleArea(p1, p2, p3) + _triangleArea(p1, p3, p4);
            // 주의: 순서가 꼬이면 면적 계산이 이상할 수 있음.
            // Convex Hull 위의 점들이므로 순서대로 선택하면 볼록 사각형이 됨.

            if (area > maxArea) {
              maxArea = area;
              bestQuad = [p1, p2, p3, p4];
            }
          }
        }
      }
    }

    return bestQuad != null ? orderCorners(bestQuad) : null;
  }

  static double _triangleArea(IntPoint a, IntPoint b, IntPoint c) {
    return 0.5 *
        ((a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y)).abs());
  }

  /// Convex Hull 알고리즘 (Graham Scan)
  static List<IntPoint> convexHull(List<IntPoint> points) {
    if (points.length <= 2) return points;

    // 복사본 생성 및 정렬
    final sorted = List<IntPoint>.from(points);
    sorted.sort((a, b) {
      final dx = a.x - b.x;
      return dx != 0 ? dx : a.y - b.y;
    });

    final lower = <IntPoint>[];
    for (final p in sorted) {
      while (lower.length >= 2 &&
          _cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
        lower.removeLast();
      }
      lower.add(p);
    }

    final upper = <IntPoint>[];
    for (final p in sorted.reversed) {
      while (upper.length >= 2 &&
          _cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
        upper.removeLast();
      }
      upper.add(p);
    }

    lower.removeLast();
    upper.removeLast();
    return [...lower, ...upper];
  }

  static int _cross(IntPoint a, IntPoint b, IntPoint c) {
    final abx = b.x - a.x;
    final aby = b.y - a.y;
    final acx = c.x - a.x;
    final acy = c.y - a.y;
    return abx * acy - aby * acx;
  }

  /// 꼭지점을 Top-Left부터 시계방향으로 정렬
  static List<IntPoint> orderCorners(List<IntPoint> corners) {
    if (corners.length != 4) {
      return corners;
    }

    // 무게중심 계산
    var cx = 0.0;
    var cy = 0.0;
    for (final corner in corners) {
      cx += corner.x;
      cy += corner.y;
    }
    cx /= 4;
    cy /= 4;

    // 무게중심 기준 각도로 정렬 (-PI ~ PI)
    final sorted = List<IntPoint>.from(corners);
    sorted.sort((a, b) {
      final angleA = math.atan2(a.y - cy, a.x - cx);
      final angleB = math.atan2(b.y - cy, b.x - cx);
      return angleA.compareTo(angleB);
    });

    // atan2는 (-PI, PI] 범위를 가짐.
    // 일반적인 화면 좌표계(y가 아래로 증가)에서:
    // Top-Left: -3PI/4 (약 -135도)
    // Top-Right: -PI/4 (약 -45도)
    // Bottom-Right: PI/4 (약 45도)
    // Bottom-Left: 3PI/4 (약 135도)

    // 하지만 문서가 회전되어 있을 수 있으므로,
    // 가장 상단(y가 작은) 점들 중 가장 왼쪽(x가 작은) 점을 Top-Left로 정의하는 것이 안전함.
    // 혹은 합(x+y)과 차(x-y)를 이용하는 방식도 있음.

    // 여기서는 정렬된 점들 중에서 "Top-Left"를 찾아서 순서를 맞춤.
    // Top-Left는 보통 x+y가 가장 작음.

    int tlIndex = 0;
    int minSum = sorted[0].x + sorted[0].y;

    for (int i = 1; i < 4; i++) {
      final sum = sorted[i].x + sorted[i].y;
      if (sum < minSum) {
        minSum = sum;
        tlIndex = i;
      }
    }

    return [
      sorted[tlIndex],
      sorted[(tlIndex + 1) % 4],
      sorted[(tlIndex + 2) % 4],
      sorted[(tlIndex + 3) % 4],
    ];
  }

  /// 원근 변환 적용
  static img.Image? warpPerspective(
    img.Image source,
    List<Corner> corners, {
    int? outputWidth,
    int? outputHeight,
  }) {
    if (corners.length != 4) {
      return null;
    }

    // Corner 정렬 (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
    // 입력된 corners가 이미 정렬되어 있다고 가정하지만, 안전을 위해 다시 확인 가능.
    // 여기서는 입력 순서를 신뢰하거나, 내부적으로 다시 정렬.
    // warpPerspective를 부르기 전에 이미 orderCorners를 거쳤을 것이므로 그대로 사용.

    final tl = corners[0];
    final tr = corners[1];
    final br = corners[2];
    final bl = corners[3];

    final width = outputWidth ?? _calculateOutputWidth(tl, tr, br, bl);
    final height = outputHeight ?? _calculateOutputHeight(tl, tr, br, bl);

    if (width <= 0 || height <= 0) {
      return null;
    }

    final destination = [
      Corner(0, 0),
      Corner(width - 1.0, 0),
      Corner(width - 1.0, height - 1.0),
      Corner(0, height - 1.0),
    ];

    final homography = _computeHomography(src: corners, dst: destination);

    if (homography == null) {
      return null;
    }

    final output = img.Image(width: width, height: height);

    // 역방향 매핑 (Inverse Mapping)
    // 출력 이미지의 각 픽셀 (x, y)에 대해 입력 이미지의 좌표 (u, v)를 계산
    // 이를 위해 Homography의 역행렬이 필요하거나,
    // _computeHomography를 (dst -> src)로 계산해야 함.
    // 기존 코드는 src -> dst 호모그래피를 구하고 _mapPoint로 역변환(?)을 시도하는 것처럼 보였으나,
    // 일반적으로 warpPerspective는 dst 픽셀을 순회하며 src 픽셀을 가져오므로
    // dst -> src 매핑 행렬(H_inv)이 필요함.
    // 기존 코드를 보면 _mapPoint 공식이 H_inv를 적용하는 것인지 확인 필요.
    // 기존 코드: srcX = (h0*x + h1*y + h2)/denom ...
    // 이는 dst(x,y) -> src(u,v) 변환식임. 즉, 여기서 구한 homography는 dst -> src 변환 행렬이어야 함.
    // 따라서 _computeHomography 호출 시 src와 dst를 바꿔서 넣어야 함.

    // 수정: dst -> src 변환을 위해 src=destination, dst=corners 로 호출
    final invHomography = _computeHomography(src: destination, dst: corners);

    if (invHomography == null) return null;

    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        final mapped = _mapPoint(invHomography, x.toDouble(), y.toDouble());
        if (mapped == null) continue;

        final color = _getPixelBilinear(source, mapped.$1, mapped.$2);
        if (color == null) continue;

        output.setPixelRgba(
          x,
          y,
          color.r.toInt(),
          color.g.toInt(),
          color.b.toInt(),
          color.a.toInt(),
        );
      }
    }

    return output;
  }

  static int _calculateOutputWidth(Corner tl, Corner tr, Corner br, Corner bl) {
    final widthA = _distance(tl, tr);
    final widthB = _distance(bl, br);
    return math.max(widthA, widthB).round();
  }

  static int _calculateOutputHeight(
    Corner tl,
    Corner tr,
    Corner br,
    Corner bl,
  ) {
    final heightA = _distance(tl, bl);
    final heightB = _distance(tr, br);
    return math.max(heightA, heightB).round();
  }

  static double _distance(Corner a, Corner b) {
    final dx = b.x - a.x;
    final dy = b.y - a.y;
    return math.sqrt(dx * dx + dy * dy);
  }

  static (double, double)? _mapPoint(List<double> h, double x, double y) {
    final denom = h[6] * x + h[7] * y + 1.0;
    if (denom.abs() < 1e-6) return null;
    final u = (h[0] * x + h[1] * y + h[2]) / denom;
    final v = (h[3] * x + h[4] * y + h[5]) / denom;
    return (u, v);
  }

  static List<double>? _computeHomography({
    required List<Corner> src,
    required List<Corner> dst,
  }) {
    // Gaussian elimination to solve for homography matrix
    // ... (Same implementation as before, adapted for Corner class)
    if (src.length != 4 || dst.length != 4) return null;

    final matrix = List.generate(8, (_) => List<double>.filled(8, 0));
    final vector = List<double>.filled(8, 0.0);

    for (var i = 0; i < 4; i++) {
      final x = src[i].x;
      final y = src[i].y;
      final u = dst[i].x;
      final v = dst[i].y;

      final row = i * 2;
      matrix[row][0] = x;
      matrix[row][1] = y;
      matrix[row][2] = 1;
      matrix[row][6] = -x * u;
      matrix[row][7] = -y * u;
      vector[row] = u;

      matrix[row + 1][3] = x;
      matrix[row + 1][4] = y;
      matrix[row + 1][5] = 1;
      matrix[row + 1][6] = -x * v;
      matrix[row + 1][7] = -y * v;
      vector[row + 1] = v;
    }

    final solution = _solveLinearSystem(matrix, vector);
    if (solution == null) return null;

    return [...solution, 1.0];
  }

  static List<double>? _solveLinearSystem(
    List<List<double>> matrix,
    List<double> vector,
  ) {
    final n = vector.length;
    for (var i = 0; i < n; i++) {
      var maxRow = i;
      var maxVal = matrix[i][i].abs();
      for (var k = i + 1; k < n; k++) {
        final value = matrix[k][i].abs();
        if (value > maxVal) {
          maxVal = value;
          maxRow = k;
        }
      }

      if (maxVal < 1e-9) return null;

      if (maxRow != i) {
        final tempRow = matrix[i];
        matrix[i] = matrix[maxRow];
        matrix[maxRow] = tempRow;
        final tempVal = vector[i];
        vector[i] = vector[maxRow];
        vector[maxRow] = tempVal;
      }

      for (var k = i + 1; k < n; k++) {
        final factor = matrix[k][i] / matrix[i][i];
        for (var j = i; j < n; j++) {
          matrix[k][j] -= factor * matrix[i][j];
        }
        vector[k] -= factor * vector[i];
      }
    }

    final solution = List<double>.filled(n, 0);
    for (var i = n - 1; i >= 0; i--) {
      var sum = vector[i];
      for (var j = i + 1; j < n; j++) {
        sum -= matrix[i][j] * solution[j];
      }
      final pivot = matrix[i][i];
      if (pivot.abs() < 1e-9) return null;
      solution[i] = sum / pivot;
    }
    return solution;
  }

  static img.ColorRgba8? _getPixelBilinear(
    img.Image image,
    double x,
    double y,
  ) {
    if (x < 0 || y < 0 || x >= image.width - 1 || y >= image.height - 1) {
      // Edge handling
      final ix = x.round().clamp(0, image.width - 1);
      final iy = y.round().clamp(0, image.height - 1);
      final pixel = image.getPixel(ix, iy);
      return img.ColorRgba8(
        pixel.r.toInt(),
        pixel.g.toInt(),
        pixel.b.toInt(),
        pixel.a.toInt(),
      );
    }

    final x0 = x.floor();
    final y0 = y.floor();
    final dx = x - x0;
    final dy = y - y0;

    final p00 = image.getPixel(x0, y0);
    final p10 = image.getPixel(x0 + 1, y0);
    final p01 = image.getPixel(x0, y0 + 1);
    final p11 = image.getPixel(x0 + 1, y0 + 1);

    int interp(num c00, num c10, num c01, num c11) {
      return ((c00 * (1 - dx) * (1 - dy)) +
              (c10 * dx * (1 - dy)) +
              (c01 * (1 - dx) * dy) +
              (c11 * dx * dy))
          .round();
    }

    return img.ColorRgba8(
      interp(p00.r, p10.r, p01.r, p11.r),
      interp(p00.g, p10.g, p01.g, p11.g),
      interp(p00.b, p10.b, p01.b, p11.b),
      interp(p00.a, p10.a, p01.a, p11.a),
    );
  }

  /// 사각형이 평행사변형인지 확인
  static bool checkParallelQuadrilateral(List<IntPoint>? corners) {
    if (corners == null || corners.length != 4) {
      return false;
    }

    // Check if opposite sides are parallel
    final threshold = 0.2; // Angle tolerance in radians (~11 degrees)

    // Calculate vectors for opposite sides
    final side1 = _vectorFrom(corners[0], corners[1]);
    final side2 = _vectorFrom(corners[1], corners[2]);
    final side3 = _vectorFrom(corners[2], corners[3]);
    final side4 = _vectorFrom(corners[3], corners[0]);

    // Check if side1 is parallel to side3
    final angle1 = _angleBetweenVectors(side1, side3);
    final parallel1 =
        angle1.abs() < threshold || (math.pi - angle1).abs() < threshold;

    // Check if side2 is parallel to side4
    final angle2 = _angleBetweenVectors(side2, side4);
    final parallel2 =
        angle2.abs() < threshold || (math.pi - angle2).abs() < threshold;

    return parallel1 && parallel2;
  }

  static List<double> _vectorFrom(IntPoint from, IntPoint to) {
    return [(to.x - from.x).toDouble(), (to.y - from.y).toDouble()];
  }

  static double _angleBetweenVectors(List<double> v1, List<double> v2) {
    final dotProduct = v1[0] * v2[0] + v1[1] * v2[1];
    final magnitude1 = math.sqrt(v1[0] * v1[0] + v1[1] * v1[1]);
    final magnitude2 = math.sqrt(v2[0] * v2[0] + v2[1] * v2[1]);

    if (magnitude1 == 0 || magnitude2 == 0) {
      return 0;
    }

    final cosAngle = dotProduct / (magnitude1 * magnitude2);
    return math.acos(cosAngle.clamp(-1.0, 1.0));
  }

  /// 마스크에서 유의미한 점들 추출
  /// Connected Component Analysis를 사용하여 가장 큰 영역의 점들 반환
  static List<IntPoint>? extractSignificantPoints(
    List<int> mask,
    int width,
    int height,
  ) {
    final visited = List<bool>.filled(mask.length, false);
    final components = <List<IntPoint>>[];

    final directions = const [
      IntPoint(1, 0),
      IntPoint(-1, 0),
      IntPoint(0, 1),
      IntPoint(0, -1),
    ];

    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        final index = y * width + x;
        if (mask[index] == 0 || visited[index]) {
          continue;
        }
        final queue = <int>[index];
        final component = <IntPoint>[];
        visited[index] = true;
        while (queue.isNotEmpty) {
          final current = queue.removeLast();
          final cx = current % width;
          final cy = current ~/ width;
          component.add(IntPoint(cx, cy));
          for (final dir in directions) {
            final nx = cx + dir.x;
            final ny = cy + dir.y;
            if (nx < 0 || ny < 0 || nx >= width || ny >= height) {
              continue;
            }
            final nIndex = ny * width + nx;
            if (mask[nIndex] == 0 || visited[nIndex]) {
              continue;
            }
            visited[nIndex] = true;
            queue.add(nIndex);
          }
        }
        components.add(component);
      }
    }

    if (components.isEmpty) {
      return null;
    }

    // 가장 큰 컴포넌트만 사용하여 노이즈 제거
    components.sort((a, b) => b.length.compareTo(a.length));
    final largest = components.first;

    // 너무 작은 컴포넌트는 무시
    const minPoints = 50;
    if (largest.length < minPoints) {
      return null;
    }

    return largest;
  }

  /// 세그먼트 영역 비율 계산
  static double calculateSegmentAreaRatio(List<int> mask) {
    if (mask.isEmpty) {
      return 0.0;
    }
    final foregroundPixels = mask.where((pixel) => pixel > 0).length;
    return foregroundPixels / mask.length;
  }
}
