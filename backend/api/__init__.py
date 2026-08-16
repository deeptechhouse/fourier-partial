"""HTTP and WebSocket interface layer.

This layer talks to the outside world, so it owns validation, rate limiting,
and error sanitisation. It calls services; it never touches DSP directly.
"""
