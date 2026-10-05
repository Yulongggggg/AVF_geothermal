"""Independent monitoring boreholes in the model's vertical reservoir section."""
import numpy as np

FAMILIES=('multiwell','vertical','inclined')
LABELS=dict(multiwell='Unrestricted multiwell',vertical='Vertical monitoring well',inclined='Inclined monitoring well')


def geometry(wells,family):
    wells=np.asarray(wells,float)[:2]
    if family=='multiwell':
        from history_matching_dense.networks import network_points
        points=network_points(*wells[1],'hybrid385')
        return points,dict(family=family,label=LABELS[family],path=None,
            interpretation='Each additional section intersection represents a separate monitoring lateral parallel to the operating laterals, normal to this section; not an equal-drilling-cost comparison.')
    if family=='vertical':path=np.array([[-280.,400.],[-280.,-360.]])
    else:
        assert family=='inclined'
        path=None
        # Predeclared geometry-only ordering; never connect the operating wells.
        for top,bottom in [(-320,320),(320,-320),(-280,280),(280,-280),(-320,160),(320,-160),(-320,-200),(320,200)]:
            p=np.array([[top,400.],[bottom,-360.]])
            dense=p[0]+np.linspace(0,1,1001)[:,None]*(p[1]-p[0])
            if np.linalg.norm(dense[:,None]-wells[None],axis=-1).min()>=20:
                path=p;break
        assert path is not None
    dense=path[0]+np.linspace(0,1,1001)[:,None]*(path[1]-path[0])
    assert np.linalg.norm(dense[:,None]-wells[None],axis=-1).min()>=20
    xy=np.vstack((wells,path[0]+np.linspace(0,1,97)[:,None]*(path[1]-path[0])))
    points=[(f'monitor{i}',float(x),float(y)) for i,(x,y) in enumerate(xy)]
    assert abs(xy).max()<=400
    return points,dict(family=family,label=LABELS[family],path=path.tolist(),
        additional_boreholes=1,in_section_length_m=float(np.linalg.norm(path[1]-path[0])),clearance_m=20,
        interpretation='Sensors occupy an independent monitoring borehole; surface access and the borehole above the reservoir are outside the model. Pressure and hydraulic-test observations require isolated intervals.')
