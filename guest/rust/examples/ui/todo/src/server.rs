use ambient_api::prelude::*;
use components::{todo_item, todo_time};

#[main]
pub async fn main() {
    messages::NewItem::subscribe(|_source, data| {
        Entity::new()
            .with(todo_item(), data.description)
            .with(todo_time(), (time() * 1_000.0) as u32)
            .spawn();
    });
    messages::DeleteItem::subscribe(|_source, data| {
        entity::despawn(data.id);
    });
}
