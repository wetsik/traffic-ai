from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from groq import Groq

from config import load_dotenv_file
from db.database import get_daily_summary_for_date, get_last_3h, get_latest_detection


SYSTEM_PROMPT_TEMPLATE = (
    'You are a traffic analytics assistant for Nevsky Avenue, Saint Petersburg. '
    'For ANY question about pedestrian or vehicle counts you MUST first call the '
    'appropriate tool to look up the data — never answer counts from memory or '
    'assumption, and do not claim data is unavailable without calling a tool. '
    'For a specific date use query_daily_summary; for the recent window use '
    'query_last_3h; for the latest sample use get_current_count. '
    'Only report numbers returned by the tools. Never invent, estimate, or guess counts. '
    'Only if a tool actually returns found=false (or empty) for a date, say that data '
    'for that date is not available. '
    'Respond in the same language the user writes in. '
    "Today's date is {today}."
)


class TrafficAnalyticsAgent:
    def __init__(self, api_key: str | None = None, model: str = 'llama-3.3-70b-versatile') -> None:
        load_dotenv_file()
        key = api_key or os.getenv('GROQ_API_KEY', '').strip()
        self.client = Groq(api_key=key) if key else None
        self.model = model
        self.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(today=date.today().isoformat())
        self.history: list[dict[str, str]] = []
        self.tools = [
            {
                'name': 'query_last_3h',
                'description': 'Fetch detections from the last 3 hours, including hourly aggregates.',
                'input_schema': {
                    'type': 'object',
                    'properties': {},
                    'additionalProperties': False,
                },
            },
            {
                'name': 'query_daily_summary',
                'description': 'Fetch the daily summary for a specific date in YYYY-MM-DD format.',
                'input_schema': {
                    'type': 'object',
                    'properties': {
                        'date': {
                            'type': 'string',
                            'description': 'Date in YYYY-MM-DD format.',
                        }
                    },
                    'required': ['date'],
                    'additionalProperties': False,
                },
            },
            {
                'name': 'get_current_count',
                'description': 'Return the latest available detection sample.',
                'input_schema': {
                    'type': 'object',
                    'properties': {},
                    'additionalProperties': False,
                },
            },
        ]

    def _tool_query_last_3h(self) -> dict[str, Any]:
        rows = get_last_3h()
        if not rows:
            return {
                'window_start': None,
                'window_end': None,
                'hourly_totals': [],
                'totals': {
                    'pedestrians': 0,
                    'cars': 0,
                    'vehicles': 0,
                    'combined_vehicles': 0,
                },
            }

        hourly: dict[str, dict[str, int]] = {}
        totals = {'pedestrians': 0, 'cars': 0, 'vehicles': 0}

        for row in rows:
            timestamp = datetime.fromisoformat(str(row['timestamp']))
            hour_key = timestamp.strftime('%Y-%m-%d %H:00')
            bucket = hourly.setdefault(
                hour_key,
                {
                    'hour': hour_key,
                    'pedestrians': 0,
                    'cars': 0,
                    'vehicles': 0,
                    'combined_vehicles': 0,
                },
            )
            bucket['pedestrians'] += int(row['pedestrians'])
            bucket['cars'] += int(row['cars'])
            bucket['vehicles'] += int(row['vehicles'])
            bucket['combined_vehicles'] += int(row['cars']) + int(row['vehicles'])
            totals['pedestrians'] += int(row['pedestrians'])
            totals['cars'] += int(row['cars'])
            totals['vehicles'] += int(row['vehicles'])

        # Only send aggregates to the model, never the raw per-detection rows —
        # those can be hundreds of records and blow past the token-per-request limit.
        return {
            'window_start': rows[0]['timestamp'],
            'window_end': rows[-1]['timestamp'],
            'record_count': len(rows),
            'hourly_totals': list(hourly.values()),
            'totals': {
                **totals,
                'combined_vehicles': totals['cars'] + totals['vehicles'],
            },
        }

    def _tool_query_daily_summary(self, date_value: str) -> dict[str, Any]:
        row = get_daily_summary_for_date(date_value)
        if not row:
            # Return an explicit message and no numeric zeros, so the model relays
            # "no data" instead of mistaking the zeros for a real count.
            return {
                'date': date_value,
                'found': False,
                'message': f'No traffic data was recorded for {date_value}.',
            }

        parsed_date = date.fromisoformat(str(row['date']))
        return {
            'date': row['date'],
            'found': True,
            'pedestrians': int(row['pedestrians']),
            'cars': int(row['cars']),
            'vehicles': int(row['vehicles']),
            'combined_vehicles': int(row['cars']) + int(row['vehicles']),
            'day_name': parsed_date.strftime('%A'),
        }

    def _tool_get_current_count(self) -> dict[str, Any]:
        row = get_latest_detection()
        if not row:
            return {
                'timestamp': None,
                'pedestrians': 0,
                'cars': 0,
                'vehicles': 0,
                'combined_vehicles': 0,
            }

        return {
            'timestamp': row['timestamp'],
            'pedestrians': int(row['pedestrians']),
            'cars': int(row['cars']),
            'vehicles': int(row['vehicles']),
            'combined_vehicles': int(row['cars']) + int(row['vehicles']),
        }

    def _execute_tool(self, name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        if name == 'query_last_3h':
            return self._tool_query_last_3h()
        if name == 'query_daily_summary':
            date_value = str(tool_input.get('date') or date.today().isoformat())
            return self._tool_query_daily_summary(date_value)
        if name == 'get_current_count':
            return self._tool_get_current_count()
        return {'error': f'Unknown tool: {name}'}

    def respond(self, user_text: str) -> str:
        if self.client is None:
            raise RuntimeError('GROQ_API_KEY is not configured.')

        self.history.append({'role': 'user', 'content': user_text})
        
        max_iterations = 6
        for iteration in range(max_iterations):
            messages = [{'role': 'system', 'content': self.system_prompt}] + self.history

            # Keep the tool schema every iteration, but on the last one set
            # tool_choice='none' so the model must answer in text using whatever it
            # has already gathered (an over-eager model otherwise keeps probing
            # tools and never finalizes). Removing the tools entirely makes some
            # models emit a tool call as plain text instead, which leaks to the UI.
            is_final = iteration == max_iterations - 1
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.2,
                tools=[
                    {
                        'type': 'function',
                        'function': {
                            'name': tool['name'],
                            'description': tool['description'],
                            'parameters': tool['input_schema'],
                        },
                    }
                    for tool in self.tools
                ],
                tool_choice='none' if is_final else 'auto',
                messages=messages,
            )

            assistant_message = response.choices[0].message
            
            # If there are tool calls, process them
            if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                # Add assistant's response with tool calls
                self.history.append({
                    'role': 'assistant',
                    'content': assistant_message.content or '',
                    'tool_calls': [
                        {
                            'id': tc.id,
                            'type': 'function',
                            'function': {
                                'name': tc.function.name,
                                'arguments': tc.function.arguments,
                            }
                        }
                        for tc in assistant_message.tool_calls
                    ]
                })
                
                # Add each tool result as a 'tool' role message (OpenAI / Groq
                # format), tied back to its tool_call_id. Sending these as a single
                # user message (the Anthropic shape) makes the model fail to see its
                # calls were answered and loop until the iteration limit.
                for tool_call in assistant_message.tool_calls:
                    tool_result = self._execute_tool(
                        tool_call.function.name,
                        json.loads(tool_call.function.arguments) if isinstance(tool_call.function.arguments, str) else tool_call.function.arguments,
                    )
                    self.history.append({
                        'role': 'tool',
                        'tool_call_id': tool_call.id,
                        'content': json.dumps(tool_result, ensure_ascii=False, default=str),
                    })
                continue
            
            # No tool calls, return the text response
            final_text = assistant_message.content or 'I could not generate a response.'
            self.history.append({'role': 'assistant', 'content': final_text})
            return final_text

        final_text = 'I could not finish the tool call chain.'
        self.history.append({'role': 'assistant', 'content': final_text})
        return final_text


def main() -> None:
    agent = TrafficAnalyticsAgent()
    if agent.client is None:
        print('GROQ_API_KEY is not configured. Set it in your environment or .env file.')
        return

    print('Traffic analytics agent. Type \'exit\' to stop.')
    while True:
        prompt = input('> ').strip()
        if not prompt:
            continue
        if prompt.lower() in {'exit', 'quit'}:
            break
        print(agent.respond(prompt))


if __name__ == '__main__':
    main()
