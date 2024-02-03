#----------------------------------------------------------------
# Generated CMake target import file for configuration "MinSizeRel".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "BLAKE3::blake3" for configuration "MinSizeRel"
set_property(TARGET BLAKE3::blake3 APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(BLAKE3::blake3 PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_MINSIZEREL "ASM_MASM;C"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/blake3.lib"
  )

list(APPEND _cmake_import_check_targets BLAKE3::blake3 )
list(APPEND _cmake_import_check_files_for_BLAKE3::blake3 "${_IMPORT_PREFIX}/lib/blake3.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
