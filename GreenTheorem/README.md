### The area of a shape defined by a series of curve points, using Green's theorem.
```
def compute_area(curve_pts):
    area = 0.0

    for i in range(1, len(curve_pts)):
        x1, y1 = curve_pts[i-1]
        x2, y2 = curve_pts[i]

        area += x1 * y2 - x2 * y1

    return abs(area / 2)

```
