import openvino as ov
import openvino.properties as props

core = ov.Core()
core.available_devices

device = "NPU"
core.get_property(device, props.device.full_name)
# core.get_property('GPU', props.device.full_name)

print(f"{device} SUPPORTED_PROPERTIES:\n")
supported_properties = core.get_property(device, props.supported_properties)
indent = len(max(supported_properties, key=len))

for property_key in supported_properties:
    if property_key not in (
        "SUPPORTED_METRICS",
        "SUPPORTED_CONFIG_KEYS",
        "SUPPORTED_PROPERTIES",
    ):
        try:
            property_val = core.get_property(device, property_key)
        except TypeError:
            property_val = "UNSUPPORTED TYPE"
        print(f"{property_key:<{indent}}: {property_val}")
