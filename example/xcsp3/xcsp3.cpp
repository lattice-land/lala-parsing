// Copyright 2022 Pierre Talbot

#include "battery/allocator.hpp"
#include "lala/XCSP3_parser.hpp"

using namespace lala;

int main(int argc, char *argv[]) {
  if(argc < 2) {
    printf("usage: %s <XCSP3-filename.xml>\n", argv[0]);
    return 1;
  }
  auto f = parse_xcsp3<battery::standard_allocator>(argv[1]);
  f->print(false);
  return 0;
}
