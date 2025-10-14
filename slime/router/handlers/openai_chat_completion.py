"""
Simplified OpenAI Chat Completion API implementation for Slime Router.

This module provides 100% OpenAI-compatible Chat Completion API while leveraging
Slime Router's Radix Cache for optimal performance in multi-turn conversations.

Key Features:
- Full OpenAI API compatibility (text in/out)
- Unified flow: messages → generate → OpenAI format
- Radix Tree Middleware integration for automatic caching
- Streaming and non-streaming support
- Simplified architecture with minimal abstraction

Architecture:
- Detect RadixTreeMiddleware presence
- Use query_cache_by_messages_template for semantic caching
- Forward to /generate endpoint for consistent processing
- Convert responses to OpenAI format
"""

import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.responses import Response

# Try to import SGLang parsers for advanced output processing
try:
    from sglang.srt.parser.reasoning_parser import ReasoningParser
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
    SGLANG_PARSERS_AVAILABLE = True
except ImportError:
    SGLANG_PARSERS_AVAILABLE = False
    ReasoningParser = None
    FunctionCallParser = None


class ChatCompletionHandler:
    """
    Simplified Chat Completion handler with unified processing flow.

    This handler automatically detects cache capability by testing the
    /retrieve_from_messages_template endpoint and uses the appropriate
    processing path:
    - With cache support: Use messages template caching
    - Without cache support: Direct proxy to SGLang
    """

    def __init__(self, router):
        """
        Initialize Chat Completion handler.

        Args:
            router: SlimeRouter instance for accessing middleware and workers
        """
        self.router = router
        self.args = router.args
        self._reasoning_parser = None  # Lazy-initialized reasoning parser
        self._function_call_parser = None  # Lazy-initialized function call parser

    async def handle_request(self, request: Request):
        """
        Handle Chat Completion request with unified flow.

        Args:
            request: FastAPI Request object

        Returns:
            Either JSON response (non-streaming) or StreamingResponse
        """
        try:
            try:
                request_data = await request.json()
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid JSON in request body: {str(e)}"
                )

            # Validate request structure
            self._validate_chat_completion_request(request_data)

            stream = request_data.get("stream", False)

            # Check if cache support is available (use router's method)
            cache_available = self.router._check_cache_availability()

            if not cache_available:
                # Direct mode: Proxy to SGLang Chat Completion API
                return await self._proxy_to_sglang_chat(request)

            # Cached mode: Direct flow with radix cache (no internal HTTP)
            return await self._handle_with_radix_cache(request_data, stream)
        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as e:
            # Catch any unexpected errors and log them
            import traceback
            error_traceback = traceback.format_exc()
            if getattr(self.args, 'verbose', False):
                print(f"[slime-router] ERROR in handle_request: {e}")
                print(f"[slime-router] Traceback:\n{error_traceback}")
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {str(e)}"
            )

    def _validate_chat_completion_request(self, request_data: dict):
        """
        Minimal validation for Chat Completion request.

        Only validate absolutely required fields. Let SGLang handle
        detailed parameter validation and return appropriate errors.

        Args:
            request_data: Parsed request data

        Raises:
            HTTPException: If basic validation fails
        """
        # Only check the absolute minimum required for OpenAI API compatibility
        if "messages" not in request_data:
            raise HTTPException(
                status_code=400,
                detail="Invalid request: 'messages' field is required"
            )

        messages = request_data["messages"]
        if not isinstance(messages, list) or len(messages) == 0:
            raise HTTPException(
                status_code=400,
                detail="Invalid request: 'messages' must be a non-empty list"
            )

        # Basic message structure check - let SGLang handle detailed validation
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid request: message at index {i} must be a dictionary"
                )

            if "role" not in message or "content" not in message:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid request: message at index {i} must have 'role' and 'content' fields"
                )

    def _fix_reasoning_content_in_response(self, content: bytes) -> bytes:
        """
        Fix SGLang reasoning parser output: move reasoning_content to content field.

        When reasoning parser is enabled, SGLang may put content in reasoning_content
        field instead of content. This ensures OpenAI compatibility by moving it back.

        Args:
            content: Response content bytes from SGLang

        Returns:
            Fixed response content bytes
        """
        try:
            response_json = json.loads(content) if content else {}

            if getattr(self.args, 'verbose', False):
                print(f"[slime-router] SGLang response has choices: {'choices' in response_json}")

            # Check if reasoning parser put content in reasoning_content
            if (isinstance(response_json, dict) and
                'choices' in response_json and
                len(response_json['choices']) > 0):

                for choice in response_json['choices']:
                    if 'message' in choice and isinstance(choice['message'], dict):
                        message = choice['message']

                        # If content is None but reasoning_content exists, merge them
                        if message.get('content') is None and 'reasoning_content' in message:
                            reasoning_content = message.get('reasoning_content', '')

                            if getattr(self.args, 'verbose', False):
                                print(f"[slime-router] Fixing content=None: using reasoning_content ({len(reasoning_content)} chars)")

                            # Put reasoning content into content field for OpenAI compatibility
                            message['content'] = reasoning_content

                # Re-serialize the modified response
                return json.dumps(response_json).encode('utf-8')

            return content

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # If parsing fails, just pass through the original response
            if getattr(self.args, 'verbose', False):
                print(f"[slime-router] Warning: Failed to process reasoning content: {e}")
            return content

    def _parse_generated_output(
        self,
        generated_text: str,
        request_data: dict
    ) -> Tuple[str, Optional[dict], Optional[str]]:
        """
        Parse generated output with SGLang parsers (reasoning + tool calls).

        This method integrates SGLang's reasoning parser and function call parser
        to process the raw model output into structured format.

        Args:
            generated_text: Raw text from model generation
            request_data: Original request data (for tools, reasoning config)

        Returns:
            Tuple of (final_text, tool_calls, reasoning_text)
            - final_text: Text content to show to user (after parsing)
            - tool_calls: Parsed tool calls (if any)
            - reasoning_text: Extracted reasoning content (if any)
        """
        if not SGLANG_PARSERS_AVAILABLE:
            # Parsers not available - return raw text
            return generated_text, None, None

        final_text = generated_text
        tool_calls = None
        reasoning_text = None

        try:
            # Step 1: Parse reasoning content (if reasoning parser configured)
            reasoning_parser_type = getattr(self.args, 'sglang_reasoning_parser', None)
            if reasoning_parser_type:
                if not self._reasoning_parser:
                    # Lazy initialize reasoning parser
                    self._reasoning_parser = ReasoningParser(
                        model_type=reasoning_parser_type,
                        stream_reasoning=False  # For non-streaming, accumulate reasoning
                    )

                # Parse reasoning
                reasoning_text, normal_text = self._reasoning_parser.parse_non_stream(final_text)
                final_text = normal_text if normal_text else final_text

            # Step 2: Parse tool calls (if tools provided)
            tools = request_data.get("tools")
            tool_call_parser_type = getattr(self.args, 'sglang_tool_call_parser', None)

            if tools and tool_call_parser_type:
                if not self._function_call_parser:
                    # Lazy initialize function call parser
                    from sglang.srt.entrypoints.openai.protocol import Tool
                    # Convert OpenAI tool format to SGLang Tool format if needed
                    sglang_tools = [Tool(**tool) if isinstance(tool, dict) else tool for tool in tools]
                    self._function_call_parser = FunctionCallParser(
                        tools=sglang_tools,
                        tool_call_parser=tool_call_parser_type
                    )

                # Parse tool calls
                remaining_text, parsed_calls = self._function_call_parser.parse_non_stream(final_text)
                if parsed_calls:
                    final_text = remaining_text
                    # Convert SGLang ToolCallItem to OpenAI tool_calls format
                    tool_calls = [
                        {
                            "id": f"call_{uuid.uuid4().hex[:24]}",
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments) if isinstance(call.arguments, dict) else call.arguments
                            }
                        }
                        for call in parsed_calls
                    ]

        except Exception as e:
            # Parser error - log and return raw text
            if getattr(self.args, 'verbose', False):
                print(f"[slime-router] Warning: Parser error, using raw output: {e}")
            return generated_text, None, None

        return final_text, tool_calls, reasoning_text

    async def _proxy_to_sglang_chat(self, request: Request):
        """
        Direct proxy mode: Forward request to SGLang Chat Completion API.

        Args:
            request: FastAPI Request object

        Returns:
            Direct response from SGLang
        """
        worker_url = await self.router._use_url()
        sglang_url = f"{worker_url}/v1/chat/completions"

        body = await request.body()
        headers = dict(request.headers)

        try:
            # Check if streaming request
            try:
                request_data = json.loads(body) if body else {}
                is_streaming = request_data.get("stream", False)
            except (json.JSONDecodeError, TypeError):
                is_streaming = False

            if is_streaming:
                # Streaming proxy
                async with self.router.client.stream(
                    "POST",
                    sglang_url,
                    content=body,
                    headers=headers,
                    timeout=None
                ) as response:
                    async def generate_chunks():
                        async for chunk in response.aiter_bytes():
                            yield chunk

                    return StreamingResponse(
                        generate_chunks(),
                        status_code=response.status_code,
                        headers=dict(response.headers)
                    )
            else:
                # Non-streaming proxy
                response = await self.router.client.request("POST", sglang_url, content=body, headers=headers)
                content = await response.aread()

                # Fix reasoning content if needed
                content = self._fix_reasoning_content_in_response(content)

                return Response(
                    content=content,
                    status_code=response.status_code,
                    headers=dict(response.headers)
                )
        finally:
            await self.router._finish_url(worker_url)

    async def _proxy_to_sglang_chat_from_data(self, request_data: dict):
        """
        Direct proxy mode: Forward request data to SGLang Chat Completion API.

        This is a helper method for when we need to proxy from parsed data instead of a Request object.

        Args:
            request_data: Parsed request data

        Returns:
            Direct response from SGLang
        """
        worker_url = await self.router._use_url()
        sglang_url = f"{worker_url}/v1/chat/completions"

        try:
            # Check if streaming request
            is_streaming = request_data.get("stream", False)

            if is_streaming:
                # Streaming proxy
                async with self.router.client.stream(
                    "POST",
                    sglang_url,
                    json=request_data,
                    timeout=None
                ) as response:
                    async def generate_chunks():
                        async for chunk in response.aiter_bytes():
                            yield chunk

                    return StreamingResponse(
                        generate_chunks(),
                        status_code=response.status_code,
                        headers=dict(response.headers)
                    )
            else:
                # Non-streaming proxy with error mapping
                try:
                    response = await self.router.client.request("POST", sglang_url, json=request_data)
                    content = await response.aread()

                    # Check for SGLang errors and map to OpenAI format
                    if response.status_code >= 400:
                        await self._handle_sglang_error(response, content)

                    # Fix reasoning content if needed
                    content = self._fix_reasoning_content_in_response(content)

                    return Response(
                        content=content,
                        status_code=response.status_code,
                        headers=dict(response.headers)
                    )
                except httpx.HTTPStatusError as e:
                    # Handle HTTP errors from SGLang
                    await self._handle_sglang_error(e.response, await e.response.aread())
                    raise
                except httpx.RequestError as e:
                    # Handle connection/network errors
                    raise HTTPException(
                        status_code=503,
                        detail="Service temporarily unavailable: Unable to reach inference backend"
                    )
        finally:
            await self.router._finish_url(worker_url)

    async def _handle_sglang_error(self, response, content):
        """
        Map SGLang errors to OpenAI-compatible error format.

        Args:
            response: HTTP response from SGLang
            content: Response content bytes
        """
        try:
            error_data = json.loads(content.decode('utf-8')) if content else {}

            # Map common SGLang errors to OpenAI format
            if response.status_code == 400:
                # Validation errors - pass through SGLang's message
                detail = error_data.get('error', error_data.get('detail', 'Invalid request parameters'))
                raise HTTPException(
                    status_code=400,
                    detail=detail
                )
            elif response.status_code == 429:
                # Rate limiting
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Please try again later."
                )
            elif response.status_code >= 500:
                # Server errors
                raise HTTPException(
                    status_code=response.status_code,
                    detail="Inference service error. Please try again later."
                )
            else:
                # Other errors
                raise HTTPException(
                    status_code=response.status_code,
                    detail=error_data.get('error', error_data.get('detail', 'Unknown error'))
                )
        except (json.JSONDecodeError, UnicodeDecodeError):
            # If we can't parse the error, return a generic message
            raise HTTPException(
                status_code=response.status_code,
                detail="Service error: Unable to process request"
            )

    async def _handle_with_radix_cache(self, request_data: dict, stream: bool):
        """
        Cached mode: Direct flow with radix cache (no internal HTTP).

        This method implements the Two-Path Architecture for cache-enabled mode:
        1. Apply chat template to get text
        2. Query radix cache to get token_ids
        3. Call SGLang /generate directly with tokens (token in, token out)
        4. Maintain radix cache with output tokens
        5. Parse output with tool call/reasoning parsers
        6. Convert to OpenAI chat.completion format

        Args:
            request_data: Parsed request data
            stream: Whether streaming is requested

        Returns:
            OpenAI-formatted response
        """
        messages = request_data.get("messages", [])
        tools = request_data.get("tools", None)

        # Step 1: Get tokenizer and radix tree from component registry
        try:
            # Get radix tree and tokenizer from router's component registry
            radix_tree = None
            tokenizer = None

            if hasattr(self.router, 'component_registry'):
                if self.router.component_registry.has("radix_tree"):
                    radix_tree = self.router.component_registry.get("radix_tree")
                if self.router.component_registry.has("tokenizer"):
                    tokenizer = self.router.component_registry.get("tokenizer")

            # Fallback to legacy router.radix_tree if available
            if not radix_tree and hasattr(self.router, 'radix_tree'):
                radix_tree = self.router.radix_tree
                if hasattr(self.router, 'component_registry') and self.router.component_registry.has("tokenizer"):
                    tokenizer = self.router.component_registry.get("tokenizer")

            if not radix_tree or not tokenizer:
                raise RuntimeError("Radix tree or tokenizer not available")

            # Step 2: Apply chat template to convert messages to text
            text = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=False
            )

            if not text or not text.strip():
                raise RuntimeError("Messages template resulted in empty text")

            # Step 3: Query radix cache to get token_ids
            token_ids, _, _, _ = await radix_tree.get_or_create_tokenization_async(text)

            if not token_ids:
                raise RuntimeError("Failed to get tokens from radix tree")

        except Exception as e:
            if getattr(self.args, 'verbose', False):
                print(f"[slime-router] Warning: Failed to get cached tokens, falling back to direct mode: {e}")
            # Fallback to direct proxy
            return await self._proxy_to_sglang_chat_from_data(request_data)

        # Step 4: Call SGLang /generate directly with tokens (token in, token out)
        sampling_params = self._build_sampling_params(request_data, stream)

        if stream:
            return await self._stream_generate_with_cache(token_ids, sampling_params, radix_tree, text)
        else:
            return await self._non_stream_generate_with_cache(token_ids, sampling_params, radix_tree, text, request_data)

    def _build_sampling_params(self, request_data: dict, stream: bool) -> dict:
        """
        Build sampling parameters for SGLang generation request.

        Args:
            request_data: Parsed request data from Chat Completion API
            stream: Whether streaming is requested

        Returns:
            Dictionary of sampling parameters compatible with SGLang
        """
        sampling_params = {
            # Core generation parameters
            "max_new_tokens": request_data.get("max_tokens", 1024),
            "temperature": request_data.get("temperature", 1.0),
            "top_p": request_data.get("top_p", 1.0),
            "top_k": request_data.get("top_k", -1),
            "min_p": request_data.get("min_p", 0.0),

            # Penalty parameters
            "frequency_penalty": request_data.get("frequency_penalty", 0.0),
            "presence_penalty": request_data.get("presence_penalty", 0.0),

            # Stop conditions
            "stop": request_data.get("stop"),
            "stop_token_ids": request_data.get("stop_token_ids"),
            "ignore_eos": request_data.get("ignore_eos"),

            # Special token handling
            "skip_special_tokens": request_data.get("skip_special_tokens"),
            "spaces_between_special_tokens": request_data.get("spaces_between_special_tokens"),
            "no_stop_trim": request_data.get("no_stop_trim"),
        }

        # Remove None values to keep request clean and avoid SGLang errors
        # Note: 'stream' parameter is NOT part of sampling_params
        # It's handled separately in the request JSON for /generate endpoint
        return {k: v for k, v in sampling_params.items() if v is not None}

    async def _non_stream_generate_with_cache(
        self, token_ids: list, sampling_params: dict, radix_tree, input_text: str, request_data: dict
    ):
        """
        Non-streaming generation with direct SGLang call and cache maintenance.

        This method implements the cache-enabled path without internal HTTP:
        1. Call SGLang /generate directly with input_ids (token in)
        2. Get output_ids from response (token out)
        3. Maintain radix cache with output tokens
        4. Convert to OpenAI chat.completion format

        Args:
            token_ids: Input token IDs from radix cache
            sampling_params: Sampling parameters for generation
            radix_tree: RadixTree instance for cache maintenance
            input_text: Original input text (for cache maintenance)
            request_data: Original request data (for model name, etc.)

        Returns:
            JSONResponse: OpenAI-formatted chat.completion response
        """
        # Get a worker URL
        worker_url = await self.router._use_url()

        try:
            # Use router's shared client for consistency
            response = await self.router.client.post(
                f"{worker_url}/generate",
                json={
                    "input_ids": token_ids,
                    "sampling_params": sampling_params,
                    "return_logprob": False,
                    "return_text_in_logprobs": False,
                },
                timeout=60.0
            )
            response.raise_for_status()
            generate_data = response.json()

        except httpx.TimeoutException:
            raise HTTPException(
                status_code=504,
                detail="Request timeout while calling SGLang worker"
            )
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"SGLang worker error: {e.response.text}"
            )
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Failed to connect to SGLang worker: {str(e)}"
            )
        finally:
            await self.router._finish_url(worker_url)

        # Extract generated text and token information
        generated_text = generate_data.get("text", "")
        output_ids = generate_data.get("meta_info", {}).get("output_token_ids", [])

        # Maintain radix cache: insert full sequence (input + output)
        if output_ids:
            try:
                # Combine input and output tokens for cache insertion
                full_text = input_text + generated_text
                await radix_tree.insert_async(full_text, token_ids + output_ids)
            except Exception as e:
                if getattr(self.args, 'verbose', False):
                    print(f"[slime-router] Warning: Failed to update radix cache: {e}")

        # Calculate token usage
        prompt_tokens = len(token_ids)
        completion_tokens = len(output_ids) if output_ids else len(generated_text.split())
        total_tokens = prompt_tokens + completion_tokens

        # Parse output with SGLang parsers (reasoning + tool calls)
        final_text, tool_calls, reasoning_text = self._parse_generated_output(
            generated_text,
            request_data
        )

        # Build message content
        message_content = {
            "role": "assistant",
            "content": final_text
        }

        # Add tool_calls if present
        if tool_calls:
            message_content["tool_calls"] = tool_calls

        # Optionally include reasoning in metadata (non-standard, for debugging)
        if reasoning_text and getattr(self.args, 'include_reasoning_in_response', False):
            message_content["reasoning"] = reasoning_text

        # Convert to OpenAI format
        openai_response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": getattr(self.args, 'model_name', 'slime-model'),
            "choices": [{
                "index": 0,
                "message": message_content,
                "finish_reason": "tool_calls" if tool_calls else "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        }

        return JSONResponse(content=openai_response)

    async def _stream_generate_with_cache(
        self, token_ids: list, sampling_params: dict, radix_tree, input_text: str
    ):
        """
        Streaming generation with direct SGLang call and cache maintenance.

        This method implements the cache-enabled streaming path without internal HTTP:
        1. Call SGLang /generate with streaming
        2. Stream chunks to client in OpenAI format
        3. Maintain radix cache with final output tokens

        TODO: Integrate streaming parsers for incremental parsing
        - Use ReasoningParser.parse_stream_chunk() for reasoning content
        - Use FunctionCallParser.parse_stream_chunk() for tool calls
        - This requires buffering and state management for partial tags

        Args:
            token_ids: Input token IDs from radix cache
            sampling_params: Sampling parameters for generation
            radix_tree: RadixTree instance for cache maintenance
            input_text: Original input text (for cache maintenance)

        Returns:
            StreamingResponse: OpenAI-formatted SSE stream
        """
        async def generate_openai_chunks():
            """Generate OpenAI-formatted SSE chunks."""
            request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            created_time = int(time.time())

            # Get a worker URL
            worker_url = await self.router._use_url()
            accumulated_text = ""
            accumulated_tokens = []

            # TODO: Initialize streaming parsers here if needed
            # streaming_reasoning_parser = ...
            # streaming_function_parser = ...

            try:
                # Call SGLang /generate with streaming
                async with self.router.client.stream(
                    "POST",
                    f"{worker_url}/generate",
                    json={
                        "input_ids": token_ids,
                        "sampling_params": {**sampling_params, "stream": True},
                        "return_logprob": False,
                    },
                    timeout=None
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if line.strip() and line.startswith("data: "):
                            chunk_data = line[6:]  # Remove "data: " prefix
                            if chunk_data == "[DONE]":
                                break

                            try:
                                chunk = json.loads(chunk_data)
                                text_delta = chunk.get("text", "")

                                if text_delta:
                                    accumulated_text += text_delta

                                    # OpenAI format chunk
                                    openai_chunk = {
                                        "id": request_id,
                                        "object": "chat.completion.chunk",
                                        "created": created_time,
                                        "model": getattr(self.args, 'model_name', 'slime-model'),
                                        "choices": [{
                                            "index": 0,
                                            "delta": {
                                                "content": text_delta
                                            },
                                            "finish_reason": None
                                        }]
                                    }
                                    yield f"data: {json.dumps(openai_chunk)}\n\n"

                                # Collect output tokens for cache maintenance
                                if "meta_info" in chunk and "output_token_ids" in chunk["meta_info"]:
                                    accumulated_tokens.extend(chunk["meta_info"]["output_token_ids"])

                            except json.JSONDecodeError:
                                continue

                # Final chunk
                final_chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": getattr(self.args, 'model_name', 'slime-model'),
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"

                # Maintain radix cache after streaming completes
                if accumulated_tokens:
                    try:
                        full_text = input_text + accumulated_text
                        await radix_tree.insert_async(full_text, token_ids + accumulated_tokens)
                    except Exception as e:
                        if getattr(self.args, 'verbose', False):
                            print(f"[slime-router] Warning: Failed to update radix cache: {e}")

            except Exception as e:
                if getattr(self.args, 'verbose', False):
                    print(f"[slime-router] Streaming error: {e}")

                error_chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": getattr(self.args, 'model_name', 'slime-model'),
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "error"
                    }],
                    "error": {
                        "message": "Streaming interrupted",
                        "type": "internal_error"
                    }
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                raise
            finally:
                await self.router._finish_url(worker_url)

        return StreamingResponse(
            generate_openai_chunks(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"}
        )



# Factory function for creating ChatCompletion handlers
def create_chat_completion_handler(router) -> ChatCompletionHandler:
    """
    Factory function to create Chat Completion handler.

    Args:
        router: SlimeRouter instance

    Returns:
        Configured Chat Completion handler
    """
    return ChatCompletionHandler(router)