// Copyright (c) 2024 and onwards The McBopomofo Authors.
//
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use,
// copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.

#include "MemoryMappedFile.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <string>
#include <utility>

namespace McBopomofo {

// Windows port: fd_ holds a CreateFileMapping handle cast to intptr; -1 = closed.
MemoryMappedFile::MemoryMappedFile(MemoryMappedFile&& other) noexcept
    : fd_(std::exchange(other.fd_, -1)),
      data_(std::exchange(other.data_, nullptr)),
      length_(std::exchange(other.length_, 0)) {}

MemoryMappedFile& MemoryMappedFile::operator=(
    MemoryMappedFile&& other) noexcept {
  close();
  fd_ = std::exchange(other.fd_, -1);
  data_ = std::exchange(other.data_, nullptr);
  length_ = std::exchange(other.length_, 0);
  return *this;
}

MemoryMappedFile::~MemoryMappedFile() { close(); }

bool MemoryMappedFile::open(const char* path) {
  if (fd_ != -1) return false;
  int wlen = MultiByteToWideChar(CP_UTF8, 0, path, -1, nullptr, 0);
  if (wlen <= 0) return false;
  std::wstring wpath(wlen, L'\0');
  MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath.data(), wlen);
  HANDLE file = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ,
                            nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,
                            nullptr);
  if (file == INVALID_HANDLE_VALUE) return false;
  LARGE_INTEGER size;
  if (!GetFileSizeEx(file, &size) || size.QuadPart == 0) {
    CloseHandle(file);
    return false;
  }
  HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0,
                                      nullptr);
  CloseHandle(file);
  if (mapping == nullptr) return false;
  void* view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
  if (view == nullptr) {
    CloseHandle(mapping);
    return false;
  }
  fd_ = static_cast<intptr_t>(reinterpret_cast<intptr_t>(mapping));
  data_ = view;
  length_ = static_cast<size_t>(size.QuadPart);
  return true;
}

void MemoryMappedFile::close() {
  if (fd_ == -1) return;
  UnmapViewOfFile(data_);
  CloseHandle(reinterpret_cast<HANDLE>(fd_));
  fd_ = -1;
  length_ = 0;
  data_ = nullptr;
}

}  // namespace McBopomofo
#else

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <utility>

namespace McBopomofo {

MemoryMappedFile::MemoryMappedFile(MemoryMappedFile&& other) noexcept
    : fd_(std::exchange(other.fd_, -1)),
      data_(std::exchange(other.data_, nullptr)),
      length_(std::exchange(other.length_, 0)) {}

MemoryMappedFile& MemoryMappedFile::operator=(
    MemoryMappedFile&& other) noexcept {
  close();
  fd_ = std::exchange(other.fd_, -1);
  data_ = std::exchange(other.data_, nullptr);
  length_ = std::exchange(other.length_, 0);
  return *this;
}

MemoryMappedFile::~MemoryMappedFile() { close(); }

bool MemoryMappedFile::open(const char* path) {
  if (fd_ != -1) {
    return false;
  }

  fd_ = ::open(path, O_RDONLY);
  if (fd_ == -1) {
    return false;
  }

  struct stat sb;
  if (fstat(fd_, &sb) == -1) {
    ::close(fd_);
    fd_ = -1;
    return false;
  }

  length_ = static_cast<size_t>(sb.st_size);

  // No need to check if length_ is 0; mmmap fails on empty files.
  data_ = mmap(nullptr, length_, PROT_READ, MAP_SHARED, fd_, 0);
  if (data_ == MAP_FAILED) {
    ::close(fd_);
    fd_ = -1;
    length_ = 0;
    data_ = nullptr;
    return false;
  }

  return true;
}

void MemoryMappedFile::close() {
  if (fd_ == -1) {
    return;
  }
  munmap(data_, length_);
  ::close(fd_);
  fd_ = -1;
  length_ = 0;
  data_ = nullptr;
}

}  // namespace McBopomofo
#endif  // _WIN32
