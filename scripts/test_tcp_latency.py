import socket
import time

def test_tcp_latency(text: str, host='127.0.0.1', port=18080):
    print(f"Sending: '{text}'")
    
    start_time = time.time()
    
    # Connect
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
            # Send (newline framing; TCP_NODELAY + long-line safe recv on server)
            s.sendall(text.encode('utf-8') + b'\n')
            
            # Receive
            data = s.recv(1024)
            result = data.decode('utf-8')
    except Exception as e:
        print(f"Error: {e}")
        return
        
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    
    print(f"Received: '{result}'")
    print(f"Total Client Round-Trip Latency: {latency_ms:.1f} ms")

if __name__ == "__main__":
    test_text = "We now have 4-month-old mice that are non-diabetic that used to be diabetic, he added."
    test_tcp_latency(test_text)
