//! The request slab: dense request state addressed by slot, with slots recycled on removal.

use slab::Slab;

use crate::request::Request;
use crate::types::RequestSlot;

/// Every live request, addressed by [`RequestSlot`]. Single-owner, preallocated to its capacity.
#[derive(Debug)]
pub struct RequestSlab {
    requests: Slab<Request>,
}

impl RequestSlab {
    /// A slab with room for `capacity` requests before it grows.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            requests: Slab::with_capacity(capacity),
        }
    }

    /// Stores `request`, returning the slot that addresses it until it is removed.
    ///
    /// # Panics
    ///
    /// Panics past `u32::MAX` live requests, which no configuration reaches.
    pub fn insert(&mut self, request: Request) -> RequestSlot {
        slot_of(self.requests.insert(request))
    }

    /// Removes and returns the request at `slot`, freeing the slot for reuse.
    ///
    /// # Panics
    ///
    /// Panics when `slot` is vacant: a slot is handed out by [`RequestSlab::insert`] and removed
    /// once, so a second removal is a bookkeeping bug.
    pub fn remove(&mut self, slot: RequestSlot) -> Request {
        assert!(
            self.requests.contains(slot.index()),
            "remove of a vacant request slot {slot:?}"
        );
        self.requests.remove(slot.index())
    }

    #[must_use]
    pub fn get(&self, slot: RequestSlot) -> Option<&Request> {
        self.requests.get(slot.index())
    }

    pub fn get_mut(&mut self, slot: RequestSlot) -> Option<&mut Request> {
        self.requests.get_mut(slot.index())
    }

    /// Live requests.
    #[must_use]
    pub fn len(&self) -> usize {
        self.requests.len()
    }

    /// Requests the slab holds before it grows. Growing moves every request it holds, so a
    /// caller that sized it for its whole population can check here that it stayed that size.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.requests.capacity()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    /// Every live request with its slot, in slot order.
    pub fn iter(&self) -> impl Iterator<Item = (RequestSlot, &Request)> {
        self.requests
            .iter()
            .map(|(key, request)| (slot_of(key), request))
    }

    /// Every live request with its slot, in slot order, mutably.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (RequestSlot, &mut Request)> {
        self.requests
            .iter_mut()
            .map(|(key, request)| (slot_of(key), request))
    }
}

fn slot_of(key: usize) -> RequestSlot {
    RequestSlot::new(u32::try_from(key).expect("request slots fit u32"))
}

#[cfg(test)]
mod tests {
    use super::RequestSlab;
    use crate::request::{
        egress, EgressReceiver, NewRequest, Priority, Request, SamplingParams, StopCriteria,
    };
    use crate::test_support::tokens;
    use crate::types::{RequestId, StepId};

    /// A request whose client's receiver is held for the test's lifetime.
    fn request(id: u64, clients: &mut Vec<EgressReceiver>) -> Request {
        let (sender, receiver) = egress();
        clients.push(receiver);
        Request::new(
            RequestId::new(id),
            NewRequest {
                prompt: vec![1],
                sampling: SamplingParams::default(),
                stop: StopCriteria {
                    max_new_tokens: tokens(1),
                    ignore_eos: false,
                },
                priority: Priority::default(),
                egress: sender,
            },
            StepId::new(0),
        )
    }

    #[test]
    fn slots_address_their_requests_and_are_recycled_after_removal() {
        let mut clients = Vec::new();
        let mut slab = RequestSlab::with_capacity(2);
        assert!(slab.is_empty());
        let first = slab.insert(request(1, &mut clients));
        let second = slab.insert(request(2, &mut clients));
        assert_ne!(first, second);
        assert_eq!(slab.len(), 2);
        assert_eq!(slab.get(first).map(Request::id), Some(RequestId::new(1)));
        assert_eq!(slab.get(second).map(Request::id), Some(RequestId::new(2)));

        let removed = slab.remove(first);
        assert_eq!(removed.id(), RequestId::new(1));
        assert!(slab.get(first).is_none());
        assert_eq!(slab.len(), 1);

        let third = slab.insert(request(3, &mut clients));
        assert_eq!(third, first, "a vacated slot is reused");
        assert_eq!(
            slab.iter()
                .map(|(_, request)| request.id().get())
                .collect::<Vec<_>>(),
            [3, 2]
        );
        slab.get_mut(third).unwrap().sequences_mut()[0].push_token(5);
        assert_eq!(slab.get(third).unwrap().sequences()[0].total(), 2);
    }

    #[test]
    #[should_panic(expected = "vacant request slot")]
    fn removing_a_vacant_slot_is_a_bookkeeping_bug() {
        let mut clients = Vec::new();
        let mut slab = RequestSlab::with_capacity(1);
        let slot = slab.insert(request(1, &mut clients));
        slab.remove(slot);
        slab.remove(slot);
    }
}
