import pytest
import mujoco
import numpy as np


def test_test_nullspace():
    
    model = mujoco.MjModel.from_xml_path(
        "models/panda/panda.xml"
    )
    
    data = mujoco.MjData(model)
    
    data.qpos[:7] = np.array([
        0.3,
        -0.5,
        0.2,
        -1.0,
        0.4,
        1.2,
        0.1
    ])
    
    mujoco.mj_forward(
        model,
        data
    )
    
    body_id = model.body(
        "hand"
    ).id
    
    jacp = np.zeros(
        (3, model.nv)
    )
    
    jacr = np.zeros(
        (3, model.nv)
    )
    
    mujoco.mj_jacBody(
        model,
        data,
        jacp,
        jacr,
        body_id
    )
    
    print()
    
    print("Position Jacobian")
    print(jacp)
    
    print()
    
    print("Rotation Jacobian")
    print(jacr)
    
    print()
    
    J = np.vstack([
        jacp,
        jacr
    ])
    
    print("Full Jacobian")
    print(J)
    
    print()
    
    print("shape =", J.shape)
    
    
    # 你刚算出来的 J_arm
    # 这里直接接上前面的代码
    
    J_arm = J[:, :7]
    
    N = np.eye(7) - np.linalg.pinv(J_arm) @ J_arm
    
    print()
    
    print("Null Space Projector")
    
    print(N)
    
    print()
    
    print("rank =", np.linalg.matrix_rank(N))
    
    print()
    
    print("rank(J_arm) =")
    print(
        np.linalg.matrix_rank(J_arm)
    )
    
    U,S,Vt = np.linalg.svd(J_arm)
    
    print(S)
    
    print(
        np.linalg.det(
            J_arm.T @ J_arm
        )
    )
    
    eigvals = np.linalg.eigvals(N)
    
    print(eigvals)
    
    print()
    
    print(
        np.linalg.norm(
            J_arm @ N
        )
    )
    null_err = np.linalg.norm(J_arm @ N)
    print("null space error =", null_err)
    assert null_err < 1e-10
if __name__ == "__main__":
    test_test_nullspace()
