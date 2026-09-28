import json
import queue
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from harness.codex_backend import AppServer, parse_decision, step, transcript_input
from harness.vlm_client import Reply, Session, ToolCall, VLMClient

TOOLS = [{"type": "function", "function": {"name": "find", "description": "Find object",
         "parameters": {"type": "object", "properties": {"label": {"type": "string"}},
                        "required": ["label"]}}}]


def decision(name="find", arguments='{"label":"block"}'):
    return json.dumps({"text": "Looking", "tool_calls": [{"name": name, "arguments_json": arguments}]})


class CodexTests(unittest.TestCase):
    def test_no_openai_client_for_subscription(self):
        with patch('openai.OpenAI', side_effect=AssertionError('API client created')):
            client = VLMClient('gpt-6-astra', api='codex', verbose=False)
            self.assertIsNone(client._client)

    def test_unknown_backend_fails(self):
        with self.assertRaises(ValueError):
            VLMClient('gpt-6-astra', api='typo')

    def test_tool_calls_keep_arguments_and_unique_ids(self):
        text, calls = parse_decision(decision(), TOOLS, 'required')
        self.assertEqual(calls[0].arguments, {'label': 'block'})
        self.assertNotEqual(calls[0].id, parse_decision(decision(), TOOLS, 'auto')[1][0].id)

    def test_invalid_decisions_never_reach_dispatch(self):
        for output, choice in [(decision('shell'), 'auto'), (decision(), 'none'),
                               (decision(arguments='[]'), 'auto'),
                               (decision(arguments='{broken'), 'auto'),
                               ('{"text":"ok","tool_calls":[]}', 'required')]:
            with self.subTest(output=output, choice=choice), self.assertRaises(ValueError):
                parse_decision(output, TOOLS, choice)

    def test_transcript_preserves_results_and_compaction(self):
        session = Session('system')
        session.entries.append({'kind': 'user', 'parts': [
            {'type': 'text', 'text': 'before'}, {'type': 'image', 'url': 'data:image/png;base64,eA=='},
            {'type': 'text', 'text': 'after'}]})
        call = ToolCall('call1', 'find', {'label': 'block'}, '{"label":"block"}')
        session.assistant(Reply('look', [call], {}, 0))
        session.tool_result('call1', '{"id":"observed_block"}')
        parts = transcript_input(session)
        self.assertEqual([p['type'] for p in parts], ['text', 'text', 'image', 'text', 'text', 'text'])
        self.assertIn('observed_block', parts[-1]['text'])
        self.assertIn('call1', parts[-1]['text'])
        session.compact(0)
        self.assertFalse(any(p['type'] == 'image' for p in transcript_input(session)))

    def test_receive_timeout(self):
        server = AppServer.__new__(AppServer)
        server.deadline = time.monotonic() + .01
        server.messages = queue.Queue()
        with self.assertRaises(TimeoutError):
            server.receive()

    def test_server_capability_requests_are_rejected(self):
        server = AppServer.__new__(AppServer)
        server.deadline = time.monotonic() + 1
        server.messages = queue.Queue()
        server.messages.put({'id': 2, 'method': 'item/tool/call'})
        sent = []
        server.send = sent.append
        with self.assertRaises(RuntimeError):
            server.receive()
        self.assertIn('error', sent[0])

    def test_protocol_success_and_failed_turn(self):
        class FakeServer:
            workdir = SimpleNamespace(name='/tmp/empty')
            account_type = 'chatgpt'
            failed = False
            def __init__(self, timeout):
                self.events = iter([
                    {'method': 'item/completed', 'params': {'threadId': 'thread', 'turnId': 'turn',
                     'item': {'type': 'agentMessage', 'phase': 'final_answer', 'text': decision()}}},
                    {'method': 'turn/completed', 'params': {'threadId': 'thread',
                     'turn': {'id': 'turn', 'status': 'failed' if self.failed else 'completed'}}}])
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def rpc(self, method, params):
                if method == 'thread/start':
                    assert params['ephemeral'] and params['environments'] == []
                    return {'thread': {'id': 'thread'}}
                assert params['outputSchema'] and params['input']
                return {'turn': {'id': 'turn'}}
            def event(self): return next(self.events)
        session = Session('system')
        session.user('Find the block')
        client = SimpleNamespace(timeout=1, model='gpt-6-astra', reasoning_effort='medium')
        with patch('harness.codex_backend.AppServer', FakeServer):
            reply = step(client, session, TOOLS, 'auto')
            self.assertEqual(reply.tool_calls[0].name, 'find')
            FakeServer.failed = True
            with self.assertRaises(RuntimeError):
                step(client, session, TOOLS, 'auto')


if __name__ == '__main__':
    unittest.main()
