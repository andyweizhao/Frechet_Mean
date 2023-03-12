# pictures done with https://pypi.org/project/hyperbolic/

from drawsvg import Drawing
from hyperbolic import euclid, util
from hyperbolic.poincare import *
import hyperbolic_mean

d = Drawing(2.1, 2.1, origin='center')
d.draw(euclid.Circle(0, 0, 1), fill='silver')

points, eucl_mean, fr_mean = hyperbolic_mean.example()

for p in points:
    d.draw(Point(*p), radius=.05, fill='blue')
d.draw(Point(*eucl_mean), radius=.05, fill='red')
d.draw(Point(*fr_mean), radius=.05, fill='green')

for p in points:
    d.draw(Line.from_points(*p, *fr_mean, segment=True), hwidth=.03)

d.set_render_size(w=400)
d.save_svg('hyperbolic_frechet_mean_example.svg')


