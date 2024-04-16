from exportedfuns import dq_dt
from zero import ZeroClient

zero_client = ZeroClient("localhost", 5559)

def echo():
    resp = zero_client.call("echo", "Hi there!")
    print(resp)

def hello():
    resp = zero_client.call("hello_world", None)
    print(resp)

def test():
    d={'u':[1,0,0,0],'dt':0.1}
    resp = zero_client.call("advance",d)
    print(resp)

if __name__ == "__main__":
    echo()
    hello()
    test()