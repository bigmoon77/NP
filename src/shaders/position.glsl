uniform float delta;
uniform sampler2D textureInitial;
uniform float resetState;

void main() {
  vec2 uv = gl_FragCoord.xy / resolution.xy;
  if (resetState > 0.5) { gl_FragColor = texture2D(textureInitial, uv); return; }
  vec3 position = texture2D(texturePosition, uv).xyz;
  vec3 velocity = texture2D(textureVelocity, uv).xyz;
  position += velocity * delta;
  gl_FragColor = vec4(position, 1.0);
}