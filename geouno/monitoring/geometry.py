"""Three monitoring access geometries in the modelled vertical reservoir section.

Source: ``operator_final_20260930/monitoring_geometry.py``.

* ``multiwell`` (Unrestricted multiwell): 385 candidate section intersections, each
  representing a separate monitoring lateral parallel to the operating wells.
* ``vertical`` (Vertical monitoring well): 97 equally spaced readings on one independent
  vertical borehole at x = -280 m, plus the two operating wells (99 candidates).
* ``inclined`` (Inclined monitoring well): the first predeclared inclined path that keeps
  20 m clearance from both operating wells, 97 readings plus the wells (99 candidates).

Candidate 0 is the injector and candidate 1 the producer in every geometry.
"""
import numpy as np

from .networks import candidate_points

FAMILIES = ('multiwell', 'vertical', 'inclined')
LABELS = dict(multiwell='Unrestricted multiwell', vertical='Vertical monitoring well',
              inclined='Inclined monitoring well')


def geometry(wells, family):
    """Candidate points for ``family`` given ``wells = [[x_inj, y_inj], [x_prod, y_prod]]``."""
    wells = np.asarray(wells, float)[:2]
    if family == 'multiwell':
        points = candidate_points(*wells[1], 'hybrid385')
        return points, dict(family=family, label=LABELS[family], path=None,
                            interpretation='Each additional section intersection represents a separate monitoring '
                                           'lateral parallel to the operating laterals, normal to this section; '
                                           'not an equal-drilling-cost comparison.')
    if family == 'vertical':
        path = np.array([[-280., 400.], [-280., -360.]])
    else:
        assert family == 'inclined'
        path = None
        # Predeclared geometry-only ordering; never connect the operating wells.
        for top, bottom in [(-320, 320), (320, -320), (-280, 280), (280, -280),
                            (-320, 160), (320, -160), (-320, -200), (320, 200)]:
            p = np.array([[top, 400.], [bottom, -360.]])
            dense = p[0] + np.linspace(0, 1, 1001)[:, None] * (p[1] - p[0])
            if np.linalg.norm(dense[:, None] - wells[None], axis=-1).min() >= 20:
                path = p
                break
        assert path is not None
    dense = path[0] + np.linspace(0, 1, 1001)[:, None] * (path[1] - path[0])
    assert np.linalg.norm(dense[:, None] - wells[None], axis=-1).min() >= 20
    xy = np.vstack((wells, path[0] + np.linspace(0, 1, 97)[:, None] * (path[1] - path[0])))
    points = [(f'monitor{i}', float(x), float(y)) for i, (x, y) in enumerate(xy)]
    assert abs(xy).max() <= 400
    return points, dict(family=family, label=LABELS[family], path=path.tolist(),
                        additional_boreholes=1, in_section_length_m=float(np.linalg.norm(path[1] - path[0])),
                        clearance_m=20,
                        interpretation='Sensors occupy an independent monitoring borehole; surface access and the '
                                       'borehole above the reservoir are outside the model. Pressure and '
                                       'hydraulic-test observations require isolated intervals.')
