#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include "NMTWrapper.h"
#include <cctype>
#include <cstdio>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <stdlib.h>
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
using namespace std;

// Link with Ws2_32.lib
#pragma comment(lib, "Ws2_32.lib")

// Dedicated port (avoids clashing with HTTP/dev servers on 8080)
#define DEFAULT_PORT "18080"
// Max UTF-8 request body (line-delimited); must match or exceed client send size
#define MAX_REQUEST_BYTES (256 * 1024)

// Read one request: UTF-8 bytes until '\n' (recommended) or EOF. Strips optional '\r'.
static string recv_request_line(SOCKET s) {
  string buf;
  buf.reserve(4096);
  char chunk[4096];
  while (buf.size() < MAX_REQUEST_BYTES) {
    int n = recv(s, chunk, static_cast<int>(sizeof(chunk)), 0);
    if (n <= 0)
      break;
    buf.append(chunk, static_cast<size_t>(n));
    if (buf.find('\n') != string::npos)
      break;
    // Legacy clients: no '\n' but full message fits in one recv (typical short lines)
    if (n < static_cast<int>(sizeof(chunk)))
      break;
  }
  size_t nl = buf.find('\n');
  if (nl != string::npos)
    buf.resize(nl);
  if (!buf.empty() && buf.back() == '\r')
    buf.pop_back();
  return buf;
}

// Browsers may still hit arbitrary ports with HTTP; do not send that to NMT.
static bool looks_like_http_request_line(const string &s) {
  auto starts_ci = [&s](const char *word) {
    size_t len = strlen(word);
    if (s.size() < len)
      return false;
    for (size_t i = 0; i < len; ++i) {
      unsigned char c = static_cast<unsigned char>(s[i]);
      unsigned char w = static_cast<unsigned char>(word[i]);
      if (std::tolower(c) != std::tolower(w))
        return false;
    }
    return true;
  };
  return starts_ci("GET ") || starts_ci("POST ") || starts_ci("PUT ") ||
         starts_ci("HEAD ") || starts_ci("DELETE ") || starts_ci("OPTIONS ") ||
         starts_ci("PATCH ") || starts_ci("CONNECT ") || starts_ci("TRACE ");
}

static void send_http_not_nmt_message(SOCKET s) {
  const char *body =
      "This TCP port is the English-to-Italian NMT server (raw UTF-8 lines), "
      "not an HTTP server. Point your HTTP client or Unity WebRequest at a "
      "different port, or use a dedicated NMT client.\r\n";
  const int body_len = static_cast<int>(strlen(body));
  char header[256];
  int hdr_len = snprintf(header, sizeof(header),
                         "HTTP/1.1 400 Bad Request\r\n"
                         "Content-Type: text/plain; charset=utf-8\r\n"
                         "Connection: close\r\n"
                         "Content-Length: %d\r\n"
                         "\r\n",
                         body_len);
  if (hdr_len > 0 && hdr_len < static_cast<int>(sizeof(header))) {
    send(s, header, hdr_len, 0);
    send(s, body, body_len, 0);
  }
}

int main() {
  // OpenBLAS + CTranslate2 + OpenMP on Windows can trigger allocator corruption
  // ("Bad memory unallocation") unless BLAS uses a single thread.
  _putenv_s("OPENBLAS_NUM_THREADS", "1");
  _putenv_s("GOTO_NUM_THREADS", "1");
  // Let CTranslate2 use several OpenMP threads for compute (like evaluate_nmt_fast.py).
  // If you see BLAS errors again, comment this out or set to "1".
  _putenv_s("OMP_NUM_THREADS", "8");

  // Force UTF-8 for Italian characters in console
  SetConsoleOutputCP(CP_UTF8);

  // 1. Initialize Winsock
  WSADATA wsaData;
  int iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
  if (iResult != 0) {
    cerr << "WSAStartup failed with error: " << iResult << endl;
    return 1;
  }

  // 2. Setup Address Info
  struct addrinfo *result = NULL;
  struct addrinfo hints;

  ZeroMemory(&hints, sizeof(hints));
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_protocol = IPPROTO_TCP;
  hints.ai_flags = AI_PASSIVE;

  iResult = getaddrinfo(NULL, DEFAULT_PORT, &hints, &result);
  if (iResult != 0) {
    cerr << "getaddrinfo failed with error: " << iResult << endl;
    WSACleanup();
    return 1;
  }

  // 3. Create Listening Socket
  SOCKET ListenSocket =
      socket(result->ai_family, result->ai_socktype, result->ai_protocol);
  if (ListenSocket == INVALID_SOCKET) {
    cerr << "socket failed with error: " << WSAGetLastError() << endl;
    freeaddrinfo(result);
    WSACleanup();
    return 1;
  }

  // 4. Bind Socket
  iResult = bind(ListenSocket, result->ai_addr, (int)result->ai_addrlen);
  if (iResult == SOCKET_ERROR) {
    cerr << "bind failed with error: " << WSAGetLastError() << endl;
    freeaddrinfo(result);
    closesocket(ListenSocket);
    WSACleanup();
    return 1;
  }

  freeaddrinfo(result);

  // 5. Listen
  iResult = listen(ListenSocket, SOMAXCONN);
  if (iResult == SOCKET_ERROR) {
    cerr << "listen failed with error: " << WSAGetLastError() << endl;
    closesocket(ListenSocket);
    WSACleanup();
    return 1;
  }

  // 6. Load NMT in parallel with accepting connections (weights must still be read from
  //    disk once; this removes the old "nothing listens until load finishes" stall).
  mutex engine_mutex;
  condition_variable engine_cv;
  unique_ptr<NMT::NMTWrapper> engine;
  bool engine_failed = false;
  exception_ptr load_exception;

  thread([&]() {
    try {
      auto eng = make_unique<NMT::NMTWrapper>("nllb_int8",
                                              "nllb_int8/sentencepiece.bpe.model");
      // One warmup decode so the first real client pays less JIT/cache cost.
      (void)eng->translate("Hi");
      lock_guard<mutex> lock(engine_mutex);
      engine = std::move(eng);
    } catch (...) {
      lock_guard<mutex> lock(engine_mutex);
      load_exception = current_exception();
      engine_failed = true;
    }
    engine_cv.notify_all();
  }).detach();

  cout << "✅ Listening on port " << DEFAULT_PORT
       << " — model loading in background (connect anytime; translate waits until "
          "ready).\n";

  // 7. Server Loop
  while (true) {
    SOCKET ClientSocket = accept(ListenSocket, NULL, NULL);
    if (ClientSocket == INVALID_SOCKET) {
      cerr << "accept failed with error: " << WSAGetLastError() << endl;
      continue;
    }

    // Low latency for small request/response (same idea as disabling Nagle in RPC clients)
    BOOL nodelay = TRUE;
    setsockopt(ClientSocket, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<char *>(&nodelay),
               sizeof(nodelay));

    // Receive English text (send UTF-8 ending with \n for framing; avoids 1024-byte truncation)
    string input = recv_request_line(ClientSocket);
    if (!input.empty()) {
      cout << "📥 Received: " << input << endl;

      if (looks_like_http_request_line(input)) {
        cout << "⚠️  Ignoring HTTP request (wrong service on this port).\n";
        send_http_not_nmt_message(ClientSocket);
        closesocket(ClientSocket);
        continue;
      }

      NMT::NMTWrapper *engine_ptr = nullptr;
      {
        unique_lock<mutex> lk(engine_mutex);
        engine_cv.wait(lk, [&] {
          return engine != nullptr || engine_failed;
        });
        if (engine_failed) {
          cerr << "❌ NMT failed to load.\n";
          try {
            if (load_exception)
              rethrow_exception(load_exception);
          } catch (const exception &e) {
            cerr << e.what() << endl;
          }
          const char *err = "Error: NMT engine not loaded.\n";
          send(ClientSocket, err, static_cast<int>(strlen(err)), 0);
          closesocket(ClientSocket);
          continue;
        }
        engine_ptr = engine.get();
      }

      string output = engine_ptr->translate(input);
      cout << "📤 Sending: " << output << endl;

      // Send back Italian response
      int iSendResult =
          send(ClientSocket, output.c_str(), (int)output.length(), 0);
      if (iSendResult == SOCKET_ERROR) {
        cerr << "send failed with error: " << WSAGetLastError() << endl;
      }
    } else {
      cout << "Connection closing or empty request..." << endl;
    }

    // Cleanup client socket
    closesocket(ClientSocket);
  }

  // Cleanup (though we have an infinite loop)
  closesocket(ListenSocket);
  WSACleanup();

  return 0;
}