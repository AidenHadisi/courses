---
name: udemy-notes
description: Opens a Udemy course in the Cursor browser, pulls every lecture transcript, article, and quiz for one section, and writes a single review note for that section. Use when the user gives a Udemy course URL or asks for notes on a course section.
disable-model-invocation: true
---

# Udemy Notes

Input: a Udemy course URL (any `/learn/` page) and a section number. If either is missing, ask.

Cursor's browser can't play Udemy video because it has no Widevine DRM. Don't try. Read the captions and article bodies through Udemy's API from inside the logged-in tab.

## 1. Open the course

1. `browser_tabs` `list`. If the course is already open, reuse that tab. Otherwise `browser_navigate` to the URL with `position: "active"`.
2. If Udemy shows a login page, ask the user to sign in, then continue.
3. `browser_lock` the tab while you work, and `unlock` it when you're done.

Run every snippet below with `browser_cdp` → `Runtime.evaluate`, `awaitPromise: true`, `returnByValue: true`.

## 2. Find the course id and section

```js
(async () => {
  const slug = location.pathname.split('/')[2];
  const { id } = await (await fetch(`/api-2.0/courses/${slug}/?fields[course]=id,title`)).json();
  const r = await (await fetch(`/api-2.0/courses/${id}/subscriber-curriculum-items/?page_size=1000&fields[lecture]=title,asset&fields[chapter]=title,object_index&fields[quiz]=title&fields[asset]=asset_type`)).json();
  return { id, items: r.results.map(x => ({ cls: x._class, id: x.id, title: x.title, idx: x.object_index, type: x.asset?.asset_type })) };
})()
```

Lectures belong to the `chapter` before them. Section N is the chapter whose `object_index` is N.

## 3. Pull the content

Set `COURSE` and `IDS` to the section's lecture ids. Batch about eight lectures per call.

```js
(async () => {
  const COURSE = 0, IDS = [];
  const strip = h => { const d = document.createElement('div'); d.innerHTML = h || ''; return d.innerText.trim(); };
  const vtt = t => [...new Set(t.split(/\n\n+/).map(b => b.split('\n').filter(l => l && l !== 'WEBVTT' && !l.includes('-->') && !/^\d+$/.test(l)).join(' ')))].filter(Boolean).join('\n');
  const out = [];
  for (const id of IDS) {
    const { title, asset: a = {} } = await (await fetch(`/api-2.0/users/me/subscribed-courses/${COURSE}/lectures/${id}/?fields[lecture]=title,asset&fields[asset]=asset_type,body,captions`)).json();
    const en = (a.captions || []).find(c => c.locale_id?.startsWith('en'));
    const text = a.body ? strip(a.body) : en ? vtt(await (await fetch(en.url)).text()) : '';
    out.push({ title, type: a.asset_type, text });
  }
  return out;
})()
```

Quiz questions, answers, and explanations come from `/api-2.0/quizzes/{quizId}/assessments/?page_size=50`. The fields are `prompt.question`, `prompt.answers`, `prompt.feedbacks`, and `correct_response`. Use them to see which distinctions the course tests. Don't copy the questions into the note.

When a response is too big, `browser_cdp` saves it to a JSON file. Pull it into one text file with a short `python3` script, then read that file in chunks. Read every lecture before you write.

## 4. Write the note

- Follow `AGENTS.md` exactly for structure, density, formatting, and the reply.
- Write one file per section: `<Course Folder>/<N>. <Section Title>.md`. Use the existing course folder, or create one named after the course.
- If a transcript misnames a service (auto-captions often do), fix the name quietly.
- Reply with one line: the path, and whether you created or appended.
