#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include "NMTWrapper.h"
#include <iostream>
#include <string>
#include <vector>
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
using namespace std;

// Link with Ws2_32.lib
#pragma comment(lib, "Ws2_32.lib")

#define DEFAULT_PORT "8080"
#define DEFAULT_BUFLEN 1024

int main() {
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

  // 6. Initialize NMT Engine
  cout << "⏳ Loading NMT Model..." << endl;
  NMT::NMTWrapper engine("nllb_int8", "nllb_int8/sentencepiece.bpe.model");
  cout << "✅ Model Loaded. Server listening on port " << DEFAULT_PORT << endl;

  // 7. Server Loop
  while (true) {
    SOCKET ClientSocket = accept(ListenSocket, NULL, NULL);
    if (ClientSocket == INVALID_SOCKET) {
      cerr << "accept failed with error: " << WSAGetLastError() << endl;
      continue;
    }

    char recvbuf[DEFAULT_BUFLEN];
    int recvbuflen = DEFAULT_BUFLEN;

    // Receive English text
    iResult = recv(ClientSocket, recvbuf, recvbuflen, 0);
    if (iResult > 0) {
      string input(recvbuf, iResult);
      cout << "📥 Received: " << input << endl;

      // Translate
      string output = engine.translate(input);
      cout << "📤 Sending: " << output << endl;

      // Send back Italian response
      int iSendResult =
          send(ClientSocket, output.c_str(), (int)output.length(), 0);
      if (iSendResult == SOCKET_ERROR) {
        cerr << "send failed with error: " << WSAGetLastError() << endl;
      }
    } else if (iResult == 0) {
      cout << "Connection closing..." << endl;
    } else {
      cerr << "recv failed with error: " << WSAGetLastError() << endl;
    }

    // Cleanup client socket
    closesocket(ClientSocket);
  }

  // Cleanup (though we have an infinite loop)
  closesocket(ListenSocket);
  WSACleanup();

  return 0;
}