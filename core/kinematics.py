import numpy as np

L1 = 0.5
L2 = 0.5


def fk(q):

    q1 = q[0]
    q2 = q[1]

    x = (
        L1 * np.cos(q1)
        + L2 * np.cos(q1 + q2)
    )

    y = (
        L1 * np.sin(q1)
        + L2 * np.sin(q1 + q2)
    )

    return np.array([x, y])

def ik(x, y):

    c2 = (
        x * x
        + y * y
        - L1 * L1
        - L2 * L2
    ) / (2 * L1 * L2)

    # 防止数值误差
    c2 = np.clip(c2, -1.0, 1.0)

    q2 = np.arccos(c2)

    q1 = (
        np.arctan2(y, x)
        - np.arctan2(
            L2 * np.sin(q2),
            L1 + L2 * np.cos(q2)
        )
    )

    return np.array([q1, q2])

def jacobian(q):

    q1 = q[0]
    q2 = q[1]

    J = np.array([

        [
            -L1*np.sin(q1)
            -L2*np.sin(q1+q2),

            -L2*np.sin(q1+q2)
        ],

        [
            L1*np.cos(q1)
            +L2*np.cos(q1+q2),

            L2*np.cos(q1+q2)
        ]
    ])

    return J

def fk_3dof(q):

    q1, q2, q3 = q

    l1 = 0.5
    l2 = 0.3
    l3 = 0.2

    x = (
        l1 * np.cos(q1)
        + l2 * np.cos(q1 + q2)
        + l3 * np.cos(q1 + q2 + q3)
    )

    y = (
        l1 * np.sin(q1)
        + l2 * np.sin(q1 + q2)
        + l3 * np.sin(q1 + q2 + q3)
    )

    return np.array([x, y])

def jacobian_3dof(q):

    q1, q2, q3 = q

    l1 = 0.5
    l2 = 0.3
    l3 = 0.2

    s1 = np.sin(q1)
    c1 = np.cos(q1)

    s12 = np.sin(q1 + q2)
    c12 = np.cos(q1 + q2)

    s123 = np.sin(q1 + q2 + q3)
    c123 = np.cos(q1 + q2 + q3)

    J = np.array([
        [
            -l1*s1 - l2*s12 - l3*s123,
            -l2*s12 - l3*s123,
            -l3*s123
        ],
        [
            l1*c1 + l2*c12 + l3*c123,
            l2*c12 + l3*c123,
            l3*c123
        ]
    ])

    return J