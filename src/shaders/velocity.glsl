uniform sampler2D textureInitial;
uniform float delta;
uniform float elasticity;
uniform float shapeStrength;
uniform float damping;
uniform float grabRadius;
uniform float grabStrength;
uniform vec3 grabCenter;
uniform vec3 grabTarget;
uniform float isGrabbing;

void main() {
  vec2 uv = gl_FragCoord.xy / resolution.xy;
  vec3 position = texture2D(texturePosition, uv).xyz;
  vec3 velocity = texture2D(textureVelocity, uv).xyz;
  vec3 restPosition = texture2D(textureInitial, uv).xyz;

  vec2 texel = 1.0 / resolution.xy;
  vec2 uvLeft = uv - vec2(texel.x, 0.0);
  vec2 uvRight = uv + vec2(texel.x, 0.0);
  vec2 uvDown = uv - vec2(0.0, texel.y);
  vec2 uvUp = uv + vec2(0.0, texel.y);

  vec3 pLeft = texture2D(texturePosition, uvLeft).xyz;
  vec3 pRight = texture2D(texturePosition, uvRight).xyz;
  vec3 pDown = texture2D(texturePosition, uvDown).xyz;
  vec3 pUp = texture2D(texturePosition, uvUp).xyz;
  vec3 rLeft = texture2D(textureInitial, uvLeft).xyz;
  vec3 rRight = texture2D(textureInitial, uvRight).xyz;
  vec3 rDown = texture2D(textureInitial, uvDown).xyz;
  vec3 rUp = texture2D(textureInitial, uvUp).xyz;

  vec3 springForce = vec3(0.0);
  vec3 deltaPosition = pLeft - position;
  float currentLength = length(deltaPosition);
  float restLength = length(rLeft - restPosition);
  if (restLength > 0.00001) springForce += normalize(deltaPosition) * (currentLength - restLength) * elasticity;

  deltaPosition = pRight - position;
  currentLength = length(deltaPosition);
  restLength = length(rRight - restPosition);
  if (restLength > 0.00001) springForce += normalize(deltaPosition) * (currentLength - restLength) * elasticity;

  deltaPosition = pDown - position;
  currentLength = length(deltaPosition);
  restLength = length(rDown - restPosition);
  if (restLength > 0.00001) springForce += normalize(deltaPosition) * (currentLength - restLength) * elasticity;

  deltaPosition = pUp - position;
  currentLength = length(deltaPosition);
  restLength = length(rUp - restPosition);
  if (restLength > 0.00001) springForce += normalize(deltaPosition) * (currentLength - restLength) * elasticity;

  vec3 acceleration = springForce + (restPosition - position) * shapeStrength;
  acceleration += vec3(0.0, -0.35, 0.0);

  if (isGrabbing > 0.5) {
    float distanceToGrab = length(position - grabCenter);
    float influence = exp(-distanceToGrab * distanceToGrab / max(grabRadius * grabRadius, 0.0001));
    acceleration += (grabTarget - position) * grabStrength * influence;
  }

  velocity += acceleration * delta;
  velocity *= exp(-damping * delta);
  gl_FragColor = vec4(velocity, 1.0);
}