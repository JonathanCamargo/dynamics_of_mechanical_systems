from exportedfuns import dq_dt
from zero import ZeroServer
import numpy as np
from typing import Optional

app = ZeroServer(port=5559)

q=[0,0,0,0,0,0]


@app.register_rpc
def echo(msg: str) -> str:
    return msg

@app.register_rpc
async def hello_world() -> str:
    return "hello world"

@app.register_rpc
def advance(msg: dict) -> list:
    global q
    #Compute qdot
    #Integrate for a timestep dt
    #Return the new state
    u=msg['u']
    dt=msg['dt']
    qdot=dq_dt(q,None,u)    
    new_q=q+qdot*dt
    q=new_q.tolist()    
    return new_q.tolist()

@app.register_rpc
def getq(msg: dict) -> list:
    global q
    return q

if __name__ == "__main__":    
    q0=[0,0,0,0,0,0]
    q=q0
    app.run(workers=1)