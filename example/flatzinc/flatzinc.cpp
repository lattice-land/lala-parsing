// Copyright 2022 Pierre Talbot

#include "flatzinc_parser.hpp"
#include "allocator.hpp"

using namespace lala;

int main(int argc, char *argv[]) {
  if(argc < 2) {
    printf("usage: %s <flatzinc-filename.fzn>\n", argv[0]);
    return 1;
  }
  auto f = parse_flatzinc<battery::StandardAllocator>(argv[1]);
  if(!f) {
    std::cerr << "Could not parse the FlatZinc input\n" << std::endl;
  }
  else {
    f->print(true, true);
  }
  return 0;
}
