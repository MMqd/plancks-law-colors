
// NOTE: Shader automatically converted from Godot Engine 4.2.stable's StandardMaterial3D.

shader_type spatial;
render_mode blend_mix,depth_draw_opaque,cull_back,diffuse_burley,specular_schlick_ggx;
uniform vec4 albedo : source_color;
uniform sampler2D texture_albedo : source_color,filter_linear_mipmap,repeat_enable;
uniform float point_size : hint_range(0,128);
uniform float roughness : hint_range(0,1);
uniform sampler2D texture_metallic : hint_default_white,filter_linear_mipmap,repeat_enable;
uniform vec4 metallic_texture_channel;
uniform sampler2D texture_roughness : hint_roughness_r,filter_linear_mipmap,repeat_enable;
uniform float specular;
uniform float metallic;
uniform vec3 uv1_scale;
uniform vec3 uv1_offset;
uniform vec3 uv2_scale;
uniform vec3 uv2_offset;

// Planck's law color sampling variables
uniform sampler2D texture_plancks_emission : source_color, hint_default_black,filter_linear,repeat_disable;
uniform float plancks_emission_energy = 0.1;
uniform float plancks_emission_temperature_celcius = 1300.0;
uniform float temperature_cooling_rate = 0.3;
uniform bool use_color_as_age = true;

// Constant for easy conversion between degrees Celcius and degrees Kelvin
const float kelvin_conversion = 273.15;

// Unmap function: retrieves the RGB color from the temperature texture
vec3 unmap_temperature(float temperature, sampler2D texture) {
	const float min_temp = 600.0;
	const float max_temp = 60000.0;

	// Early exit if temperature is below the minimum threshold
	if (temperature < min_temp) {
		return vec3(0.0);
	}

	// Inverse exponential mapping
	float normalized = log(temperature / min_temp) / log(max_temp / min_temp);

	// Clamp between 0 and 1 to avoid sampling outside the texture
	normalized = clamp(normalized, 0.0, 1.0);

	// Sample the texture at the computed UV u-coordinate
	return texture(texture, vec2(normalized, 0.0)).rgb;
}

float calculate_cooling_temperature(float time, float initial_temperature, float ambient_temperature, float cooling_rate) {
	// Compute the exponential decay factor
	float decay = exp(-cooling_rate * time);

	// Compute the temperature difference
	float temp_diff = initial_temperature - ambient_temperature;

	return temp_diff * decay + ambient_temperature;
}

void fragment() {
	vec2 base_uv = UV;
	vec4 albedo_tex = texture(texture_albedo,base_uv);

	float metallic_tex = dot(texture(texture_metallic,base_uv),metallic_texture_channel);
	METALLIC = metallic_tex * metallic;
	vec4 roughness_texture_channel = vec4(1.0,0.0,0.0,0.0);
	float roughness_tex = dot(texture(texture_roughness,base_uv),roughness_texture_channel);
	ROUGHNESS = roughness_tex * roughness;
	SPECULAR = specular;
	vec3 plancks_emission_tex = vec3(0.0);

	if (use_color_as_age) { // Particles cool over lifetime
		plancks_emission_tex = unmap_temperature(
									calculate_cooling_temperature(
											COLOR.x * 2.0, // Particle lifetime
											// * 2.0 is not physically accurate,
											// its, so particles leave smoke
											kelvin_conversion + plancks_emission_temperature_celcius,
											kelvin_conversion + 20.0, // Room temperature as ambient temperature
											temperature_cooling_rate
									),
									texture_plancks_emission
								).rgb;

	} else { // Objects require setting the temperature in code to cool thems
		plancks_emission_tex = unmap_temperature(
									kelvin_conversion + plancks_emission_temperature_celcius,
									texture_plancks_emission
								).rgb;
	}
	EMISSION = plancks_emission_tex*plancks_emission_energy*EXPOSURE;
	ALBEDO = albedo.rgb * albedo_tex.rgb;
}
