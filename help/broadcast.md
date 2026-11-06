[comment]: # (Broadcast module help)
[version]: # (0.5)

# Broadcast
[tagbroadcast]: # (broadcast)

The `broadcast` module sends events from one object to others. A `Broadcaster` keeps a list of listeners. A `Listener` receives the events.

To use the module, first import it:

    import broadcast

Mix the two classes into your own. Call `_broadcasterInit` from the broadcaster's `init` before you send anything:

    class Counter with Broadcaster {
        init() { self._broadcasterInit() }
        tick() { self.notify("tick") }
    }

    class Printer with Listener {
        receive(sender, event) { print event }
    }

    var c = Counter()
    var p = Printer()
    p.listen(c)
    c.tick() // tick

`Scene` in the `graphics` module is a `Broadcaster`. It sends a `GraphicsEvent` when the scene changes.

[showsubtopics]: # (subtopics)

## Broadcaster
[tagbroadcaster]: # (broadcaster)

A `Broadcaster` is mixed into the class that sends events:

    class Sender with Broadcaster {
        init() { self._broadcasterInit() }
    }

`_broadcasterInit` creates the listener list. Call it from `init`.

`subscribe` adds a listener, ignoring a listener that is already subscribed. `unsubscribe` removes one:

    sender.subscribe(receiver)
    sender.unsubscribe(receiver)

A `Listener` usually calls these for you through `listen` and `ignore`.

## Listener
[taglistener]: # (listener)

A `Listener` is mixed into the class that receives events. `listen` subscribes to a broadcaster, and `ignore` unsubscribes:

    receiver.listen(sender)
    receiver.ignore(sender)

Override `receive` to handle an event. It is called as `receive(sender, event)`. The default does nothing.

You can write a separate `receive` for each event class. Morpho selects the one that matches:

    class Watcher with Listener {
        receive(sender, GraphicsEventMoved event) { print event.ids }
        receive(sender, event) { }
    }

## Notify
[tagnotify]: # (notify)

`notify` delivers an event to every current listener:

    sender.notify(event)

Each listener receives `receive(sender, event)`. Changing the listener list from inside `receive` does not affect the delivery already in progress.

Outside a batch, the event can be any value. Inside a batch, it must respond to `batchKey`, so use a `BroadcastEvent`.

## BeginBatch
[tagbeginbatch]: # (beginbatch)
[tagendbatch]: # (endbatch)

`beginBatch` holds events until the matching `endBatch`, instead of delivering them immediately:

    sender.beginBatch()
    sender.notify(ev1)
    sender.notify(ev2)
    sender.endBatch()

Batches can be nested. Events are delivered when the outermost batch ends. An `endBatch` with no matching `beginBatch` does nothing.

Events that return the same `batchKey` are merged. See `BroadcastEvent`.

## BroadcastEvent
[tagbroadcastevent]: # (broadcastevent)

A `BroadcastEvent` is an event that can be merged inside a batch. By default `batchKey` returns `nil`, and each event is delivered on its own.

Override `batchKey` to return a value that identifies events of the same kind. When a queued event already has that key, the new event's `coalesce` is called with the earlier event, and the new event replaces it:

    class Tick is BroadcastEvent {
        init() { self.n = 1 }
        batchKey() { return Tick }
        coalesce(prev) { self.n = prev.n + 1 }
    }
