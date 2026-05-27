import urllib.request, json

url = 'http://localhost:6006/graphql'

def gql(query, variables=None):
    body = {'query': query}
    if variables:
        body['variables'] = variables
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'},
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=15).read())
    if resp.get('errors'):
        raise RuntimeError(resp['errors'])
    return resp['data']

data = gql('''
{
  getProjectByName(name: "autodebug") {
    spans(
      first: 3
      rootSpansOnly: true
      sort: { col: startTime, dir: desc }
    ) {
      edges {
        node {
          name
          statusCode
          statusMessage
          startTime
          latencyMs
          tokenCountTotal
          input  { value }
          output { value }
          events { name message }
          descendants(first: 60) {
            edges {
              node {
                name
                spanKind
                statusCode
                statusMessage
                latencyMs
                tokenCountTotal
                input  { value }
                output { value }
                events { name message }
              }
            }
          }
        }
      }
    }
  }
}
''')

runs = data['getProjectByName']['spans']['edges']
print(f"Latest {len(runs)} pipeline run(s)\n{'='*60}")

for run_edge in runs:
    root = run_edge['node']
    print(f"\nRun: {root['name']} | {root['startTime']} | {root['latencyMs']}ms | tokens={root['tokenCountTotal']} | status={root['statusCode']}")
    if root['statusMessage']:
        print(f"  status_msg: {root['statusMessage']}")

    for ce in root['descendants']['edges']:
        s = ce['node']
        inp = out = ""
        if s.get('input') and s['input'].get('value'):
            try:
                v = json.loads(s['input']['value'])
                if isinstance(v, list) and v:
                    inp = str(v[-1].get('content', ''))[:300]
                else:
                    inp = str(v)[:300]
            except Exception:
                inp = str(s['input']['value'])[:300]
        if s.get('output') and s['output'].get('value'):
            try:
                v = json.loads(s['output']['value'])
                out = str(v.get('content', v))[:300] if isinstance(v, dict) else str(v)[:300]
            except Exception:
                out = str(s['output']['value'])[:300]

        evts = [f"{e['name']}: {e['message']}" for e in (s.get('events') or []) if e.get('message')]
        print(f"  [{s['spanKind']}] {s['name']} | {s['latencyMs']}ms | tok={s['tokenCountTotal']} | {s['statusCode']}")
        if s['statusMessage']:
            print(f"      err: {s['statusMessage'][:300]}")
        if inp:
            print(f"      in:  {inp}")
        if out:
            print(f"      out: {out}")
        for e in evts[:2]:
            print(f"      evt: {e[:300]}")
