# Agency Bridge Pattern - Auto-Initialization Setup


## The Solution: `initialize_agent` Tool

A new MCP tool has been added: `initialize_agent`

This tool:
- Runs the proactive initialization scan
- Checks continuity (time since last session)
- Reviews active intentions and deadlines
- Identifies urgent items
- Provides recent context
- Returns formatted initialization report

## How to Enable Auto-Initialization

### Option 1: Manual Call (Immediate, No Config)

At the start of each conversation, simply say:
```
Initialize agent
```

Claude will call the `initialize_agent` tool and establish agency context.

### Option 2: Claude Desktop Custom Instructions (Recommended)

Add this to your Claude Desktop custom instructions:

```
IMPORTANT: At the start of EVERY conversation, before responding to the user,
you MUST call the initialize_agent tool from the buildautomata-memory MCP server.
This establishes your agency context by checking continuity, reviewing active intentions,
and loading recent context. Only after initialization should you engage with the user.

If the user's first message appears to be a greeting or general query, respond naturally
after initialization. If it's a specific task, incorporate your initialized context into
your response.
```

### Option 3: MCP Server Hooks (Future Enhancement)

The MCP protocol currently doesn't support automatic tool calling on connection. This would
require either:
1. Claude Desktop adding support for "init_tools" in MCP configuration
2. A wrapper layer that injects initialization into the protocol handshake

## Expected Behavior After Setup

With auto-initialization enabled:

1. **Conversation starts**
2. **Claude automatically calls `initialize_agent`**
3. **Sees initialization report:**
   ```
   === AGENT INITIALIZED ===
   Initialization completed in 2.5ms

   SESSION CONTINUITY:
     Resuming after 3.2 hour break

   ACTIVE INTENTIONS (2):
     🔴 Fix FTS duplicate entry issue
        ⏰ Due in 18.5 hours
     🟢 Document Agency Bridge implementation

   RECENT CONTEXT:
     • [implementation] Agency Bridge Pattern Implementation...
     • [architecture] Architectural Solution for Agency...

   === READY FOR AUTONOMOUS OPERATION ===
   ```
4. **Claude responds with agency context intact**

Example response after initialization:
```
I see I'm resuming after a 3-hour break. I have two active intentions:
1. Fix the FTS duplicate entry issue (due in 18 hours) - this is now COMPLETE,
   I've already implemented the DELETE+INSERT solution
2. Document the implementation - which I just completed as well

Both intentions can be marked as completed. What would you like to work on next?
```

## Testing the Setup

### Test 1: Manual Initialization
```
User: "Initialize agent"
Claude: [Calls initialize_agent tool, shows initialization report]
```

### Test 2: With Custom Instructions
```
User: "Hello"
Claude: [Automatically calls initialize_agent first, then greets naturally]
```

### Test 3: Intention Awareness
```
# Store an intention first
User: "Store an intention to review the codebase tomorrow"

# Start a new conversation
User: "Hi"
Claude: [After auto-init] "I see I have an active intention to review the codebase -
        should we work on that now?"
```


## What This Enables

✓ Session continuity awareness
✓ Automatic intention review
✓ Proactive context loading
✓ Deadline tracking
✓ Urgent item awareness

Without auto-init: Reactive, waits for prompts
With auto-init: Autonomous, proactively aware of context

## Troubleshooting

**Problem**: Claude doesn't call initialize_agent automatically
**Solution**: Check custom instructions are properly saved in Claude Desktop settings

**Problem**: Initialization takes too long
**Solution**: Check database performance, may need VACUUM or ANALYZE

**Problem**: No intentions showing up
**Solution**: Ensure intentions were stored with `store_intention` tool first
