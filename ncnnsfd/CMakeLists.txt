
find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
if(NOT OpenCV_FOUND)
    find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../src)

add_executable(mobilenetsfd_image mobilenetsfd_image.cpp)
target_link_libraries(mobilenetsfd_image ncnn ${OpenCV_LIBS})

add_executable(mobilenetsfd_webcam mobilenetsfd_webcam.cpp)
target_link_libraries(mobilenetsfd_webcam ncnn ${OpenCV_LIBS})
