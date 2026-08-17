#pragma once

#include <pybind11/pybind11.h>

namespace deep_ep {

struct Buffer;

namespace moonep {

void bind_balanced_handle(pybind11::module_& module);
void bind_balanced_buffer(pybind11::class_<Buffer>& buffer_class);

} // namespace moonep
} // namespace deep_ep
