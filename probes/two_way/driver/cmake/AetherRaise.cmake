# aether_raise: raise kernels of a C++ target into the IR with the `raise` driver.
#
#   aether_raise(TARGET <target> ENTRY <file> KERNELS <qualified names...> ROOTS <dirs...>
#                OUT <dir> RAISE <raise binary> TRACE_INCLUDE <dir> IR <raptor checkout>
#                [STD <20>])
#
# The include directories and compile definitions come from <target>. They are written, with the
# other settings, into a response file (one argument per line) when CMake generates the build
# system; the custom target <target>_raise runs `raise @<response file>` and produces
# <OUT>/<kernel>.ir and <OUT>/<kernel>.planes, with `::` in a kernel name written as `_`.
function(aether_raise)
  cmake_parse_arguments(AR "" "TARGET;ENTRY;OUT;RAISE;TRACE_INCLUDE;IR;STD"
                        "KERNELS;ROOTS" ${ARGN})
  foreach(required TARGET ENTRY OUT RAISE TRACE_INCLUDE IR KERNELS ROOTS)
    if(NOT AR_${required})
      message(FATAL_ERROR "aether_raise: ${required} is required")
    endif()
  endforeach()
  if(NOT AR_STD)
    set(AR_STD 20)
  endif()

  set(response "${CMAKE_CURRENT_BINARY_DIR}/${AR_TARGET}.raise.rsp")
  set(includes "$<TARGET_PROPERTY:${AR_TARGET},INCLUDE_DIRECTORIES>")
  set(defines "$<TARGET_PROPERTY:${AR_TARGET},COMPILE_DEFINITIONS>")
  string(REPLACE ";" "\n" kernel_lines "${AR_KERNELS}")
  string(REPLACE ";" "\n" root_lines "${AR_ROOTS}")
  file(GENERATE OUTPUT "${response}" CONTENT
"# Request for the raise driver, written by aether_raise for ${AR_TARGET}.
--entry
${AR_ENTRY}
--kernel
${kernel_lines}
--root
${root_lines}
$<$<BOOL:${includes}>:-I$<JOIN:${includes},\n-I>>
$<$<BOOL:${defines}>:-D$<JOIN:${defines},\n-D>>
--cxx
${CMAKE_CXX_COMPILER}
--std
c++${AR_STD}
--trace-include
${AR_TRACE_INCLUDE}
--ir
${AR_IR}
--out
${AR_OUT}
")

  set(outputs)
  foreach(kernel IN LISTS AR_KERNELS)
    string(REPLACE "::" "_" file "${kernel}")
    list(APPEND outputs "${AR_OUT}/${file}.ir" "${AR_OUT}/${file}.planes")
  endforeach()
  add_custom_command(
    OUTPUT ${outputs}
    COMMAND "${AR_RAISE}" "@${response}"
    DEPENDS "${response}" "${AR_ENTRY}" "${AR_RAISE}"
    COMMENT "Raising kernels of ${AR_TARGET}"
    VERBATIM)
  add_custom_target(${AR_TARGET}_raise DEPENDS ${outputs})
endfunction()
