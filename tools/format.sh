#!/bin/bash
echo "Formatting code"
find src/ -name *.c -or -name *.cpp -or -name *.hpp -or -name *.cppm | xargs clang-format -i
find test/ -name *.c -or -name *.cpp -or -name *.hpp -or -name *.cppm | xargs clang-format -i