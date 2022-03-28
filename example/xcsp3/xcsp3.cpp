// Copyright 2022 Pierre Talbot

#include "XCSP3_parser.hpp"

using namespace lala;

int main(int argc, char *argv[]) {
  if(argc < 2) {
    printf("usage: %s <XCSP3-filename.xml>\n", argv[0]);
    return 1;
  }
  auto sf = parse_xcsp3<StandardAllocator>(argv[1], 0, 1);
  sf.formula().print(false);
  return 0;
}
