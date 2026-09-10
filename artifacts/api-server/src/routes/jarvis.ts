import { Router, type IRouter } from "express";
import multer from "multer";
import { and, desc, eq, gte, lt, sql } from "drizzle-orm";
import {
  CreateConversationBody,
  CreateConversationMessageBody,
  CreateConversationMessageParams,
  CreateConversationMessageResponse,
  CreateConversationResponse,
  CreateDocumentBody,
  CreateGoalBody,
  CreateMemoryBody,
  CreateMeetingBody,
  CreateMilestoneBody,
  CreateReminderBody,
  CreateDocumentResponse,
  CreateGoalResponse,
  CreateMemoryResponse,
  CreateMeetingResponse,
  CreateMilestoneResponse,
  CreateReminderResponse,
  DeleteConversationParams,
  DeleteDocumentParams,
  DeleteGoalParams,
  DeleteMeetingParams,
  DeleteMemoryParams,
  DeleteMilestoneParams,
  DeleteReminderParams,
  GetConversationParams,
  GetConversationResponse,
  QueryDocumentsBody,
  GetDocumentParams,
  GetDocumentResponse,
  GetGoalParams,
  GetGoalResponse,
  GetMeetingParams,
  GetMeetingResponse,
  ListConversationsResponse,
  ListDocumentsResponse,
  ListGoalsResponse,
  ListMeetingsResponse,
  ListMemoriesResponse,
  ListRemindersResponse,
  QueryDocumentsResponse,
  SearchKnowledgeQueryParams,
  SearchKnowledgeResponse,
  SendChatBody,
  SendChatResponse,
  UpdateConversationBody,
  UpdateConversationParams,
  UpdateConversationResponse,
  UpdateGoalBody,
  UpdateGoalParams,
  UpdateMemoryBody,
  UpdateMemoryParams,
  UpdateMilestoneBody,
  UpdateMilestoneParams,
  UpdateReminderBody,
  UpdateReminderParams,
} from "@workspace/api-zod";
import {
  activityEventsTable,
  chatMessagesTable,
  conversationsTable,
  db,
  documentChunksTable,
  documentsTable,
  goalsTable,
  memoriesTable,
  meetingsTable,
  milestonesTable,
  remindersTable,
  xpEventsTable,
} from "@workspace/db";
import { requireUser, currentUser } from "../lib/auth";
import { generateGemini } from "../lib/ai";
import {
  awardXp,
  buildContext,
  levelForXp,
  logActivity,
  nextLevelXp,
  retrieveDocumentChunks,
  searchKnowledge,
  splitIntoChunks,
} from "../lib/jarvis";

const router: IRouter = Router();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 15 * 1024 * 1024 } });
router.use(requireUser);

async function syncGoalProgress(goalId: number): Promise<void> {
  const all = await db.select({ status: milestonesTable.status }).from(milestonesTable).where(eq(milestonesTable.goalId, goalId));
  const done = all.filter((m) => m.status === "completed").length;
  await db.update(goalsTable).set({ progress: all.length ? Math.round((done / all.length) * 100) : 0 }).where(eq(goalsTable.id, goalId));
}

router.post("/chat", async (req, res): Promise<void> => {
  const parsed = SendChatBody.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ error: parsed.error.message });
    return;
  }
  const user = currentUser(res);
  const attachedDocId = parsed.data.attachedDocumentId;
  if (attachedDocId) {
    const [doc] = await db.select().from(documentsTable).where(and(eq(documentsTable.id, attachedDocId), eq(documentsTable.userId, user.id))).limit(1);
    if (!doc) { res.status(404).json({ error: "Attached document not found" }); return; }
    const matches = await retrieveDocumentChunks(user.id, parsed.data.message, attachedDocId);
    const context = matches.map((m) => `${m.document.title}: ${m.chunk.content}`).join("\n");
    const prompt = [
      "You are JARVIS. Answer the user's question using ONLY the provided document excerpts.",
      "If the answer is not in the document, say so clearly.",
      `Document: ${doc.title}`,
      `User question: ${parsed.data.message}`,
      `Document excerpts:\n${context}`,
    ].join("\n");
    const generated = await generateGemini(prompt);
    const sources = matches.slice(0, 3).map((m) => ({ type: "document", title: m.document.title, excerpt: m.chunk.content.slice(0, 200) }));
    res.json(SendChatResponse.parse({ reply: generated ?? `I found relevant passages in ${doc.title} but could not generate a response. Here's what matched:\n${matches[0]?.chunk.content.slice(0, 500) ?? "No matching passages."}`, sources }));
    return;
  }
  const context = await buildContext(user.id, parsed.data.message);
  const contextBlocks: string[] = [];
  if (context.goals.length) contextBlocks.push(`Active goals: ${JSON.stringify(context.goals.map((item) => ({ title: item.title, progress: item.progress, deadline: item.deadline, priority: item.priority })))}`);
  if (context.reminders.length) contextBlocks.push(`Open reminders: ${JSON.stringify(context.reminders.map((item) => ({ title: item.title, dueAt: item.dueAt })))}`);
  if (context.meetings.length) contextBlocks.push(`Recent meetings: ${JSON.stringify(context.meetings.map((item) => ({ title: item.title, summary: item.summary })))}`);
  if (context.memories.length) contextBlocks.push(`Memories: ${JSON.stringify(context.memories.map((item) => item.content))}`);
  if (context.chunks.length) contextBlocks.push(`Document excerpts: ${JSON.stringify(context.chunks.map((item) => ({ title: item.document.title, excerpt: item.chunk.content })))}`);
  const prompt = [
    "You are JARVIS, a personal context assistant.",
    "Answer the user's question using ONLY the provided personal context when it is relevant to the question.",
    "For general knowledge questions unrelated to the personal context, answer using your own knowledge.",
    "When context is provided, use it to make the answer specific and practical.",
    `User question: ${parsed.data.message}`,
    ...(contextBlocks.length ? contextBlocks : ["(No personal context was retrieved for this question — answer using general knowledge)"]),
  ].join("\n");
  const generated = await generateGemini(prompt);
  const fallback = context.goals[0]
    ? `Your most relevant focus is "${context.goals[0].title}" at ${context.goals[0].progress}% progress${context.goals[0].deadline ? `, due ${context.goals[0].deadline}` : ""}. Start with the next visible milestone and keep your open reminders in view.`
    : "I don't have specific personal context for this question. Add a goal, memory, or document and I can make future answers more tailored to you.";
  const sources = [
    ...context.goals.slice(0, 2).map((item) => ({ type: "goal", title: item.title, excerpt: `${item.progress}% complete` })),
    ...context.memories.slice(0, 2).map((item) => ({ type: "memory", title: "Memory", excerpt: item.content })),
    ...context.chunks.slice(0, 2).map((item) => ({ type: "document", title: item.document.title, excerpt: item.chunk.content.slice(0, 160) })),
  ];
  res.json(SendChatResponse.parse({ reply: generated ?? fallback, sources }));
});

router.get("/conversations", async (_req, res): Promise<void> => {
  const items = await db.select().from(conversationsTable).where(eq(conversationsTable.userId, currentUser(res).id)).orderBy(desc(conversationsTable.updatedAt));
  res.json(ListConversationsResponse.parse(items));
});

router.post("/conversations", async (req, res): Promise<void> => {
  const parsed = CreateConversationBody.safeParse(req.body);
  if (!parsed.success) { res.status(400).json({ error: parsed.error.message }); return; }
  const userId = currentUser(res).id;
  const [conversation] = await db.insert(conversationsTable).values({ userId, title: parsed.data.title }).returning();
  const messages = parsed.data.messages ?? [];
  if (messages.length) {
    await db.insert(chatMessagesTable).values(messages.map((message) => ({
      conversationId: conversation.id,
      role: message.role,
      content: message.content,
      sources: message.sources ?? [],
    })));
  }
  const createdMessages = messages.length
    ? await db.select().from(chatMessagesTable).where(eq(chatMessagesTable.conversationId, conversation.id)).orderBy(chatMessagesTable.createdAt)
    : [];
  res.status(201).json(CreateConversationResponse.parse({ ...conversation, messages: createdMessages }));
});

router.get("/conversations/:id", async (req, res): Promise<void> => {
  const params = GetConversationParams.safeParse(req.params);
  if (!params.success) { res.status(400).json({ error: params.error.message }); return; }
  const [conversation] = await db.select().from(conversationsTable).where(and(eq(conversationsTable.id, params.data.id), eq(conversationsTable.userId, currentUser(res).id))).limit(1);
  if (!conversation) { res.status(404).json({ error: "Conversation not found" }); return; }
  const messages = await db.select().from(chatMessagesTable).where(eq(chatMessagesTable.conversationId, conversation.id)).orderBy(chatMessagesTable.createdAt);
  res.json(GetConversationResponse.parse({ ...conversation, messages }));
});

router.patch("/conversations/:id", async (req, res): Promise<void> => {
  const params = UpdateConversationParams.safeParse(req.params);
  const parsed = UpdateConversationBody.safeParse(req.body);
  if (!params.success || !parsed.success) { res.status(400).json({ error: "Invalid conversation update" }); return; }
  const [conversation] = await db.update(conversationsTable).set({ title: parsed.data.title, updatedAt: new Date() }).where(and(eq(conversationsTable.id, params.data.id), eq(conversationsTable.userId, currentUser(res).id))).returning();
  if (!conversation) { res.status(404).json({ error: "Conversation not found" }); return; }
  res.json(UpdateConversationResponse.parse(conversation));
});

router.delete("/conversations/:id", async (req, res): Promise<void> => {
  const params = DeleteConversationParams.safeParse(req.params);
  if (!params.success) { res.status(400).json({ error: params.error.message }); return; }
  const [conversation] = await db.delete(conversationsTable).where(and(eq(conversationsTable.id, params.data.id), eq(conversationsTable.userId, currentUser(res).id))).returning();
  if (!conversation) { res.status(404).json({ error: "Conversation not found" }); return; }
  await db.delete(chatMessagesTable).where(eq(chatMessagesTable.conversationId, conversation.id));
  res.sendStatus(204);
});

router.post("/conversations/:id/messages", async (req, res): Promise<void> => {
  const params = CreateConversationMessageParams.safeParse(req.params);
  const parsed = CreateConversationMessageBody.safeParse(req.body);
  if (!params.success || !parsed.success) { res.status(400).json({ error: "Invalid message" }); return; }
  const [conversation] = await db.select().from(conversationsTable).where(and(eq(conversationsTable.id, params.data.id), eq(conversationsTable.userId, currentUser(res).id))).limit(1);
  if (!conversation) { res.status(404).json({ error: "Conversation not found" }); return; }
  const [message] = await db.insert(chatMessagesTable).values({
    conversationId: conversation.id,
    role: parsed.data.role,
    content: parsed.data.content,
    sources: parsed.data.sources ?? [],
  }).returning();
  await db.update(conversationsTable).set({ updatedAt: new Date() }).where(eq(conversationsTable.id, conversation.id));
  res.status(201).json(CreateConversationMessageResponse.parse(message));
});

router.get("/memories", async (_req, res): Promise<void> => {
  const items = await db.select().from(memoriesTable).where(eq(memoriesTable.userId, currentUser(res).id)).orderBy(desc(memoriesTable.updatedAt));
  res.json(ListMemoriesResponse.parse(items));
});

router.post("/memories", async (req, res): Promise<void> => {
  const parsed = CreateMemoryBody.safeParse(req.body);
  if (!parsed.success) { res.status(400).json({ error: parsed.error.message }); return; }
  const userId = currentUser(res).id;
  const [item] = await db.insert(memoriesTable).values({ ...parsed.data, userId, source: "manual" }).returning();
  await Promise.all([awardXp(userId, `memory:${item.id}`, 5), logActivity(userId, "memory", "Memory added", item.content)]);
  res.status(201).json(CreateMemoryResponse.parse(item));
});

router.patch("/memories/:id", async (req, res): Promise<void> => {
  const params = UpdateMemoryParams.safeParse(req.params);
  const parsed = UpdateMemoryBody.safeParse(req.body);
  if (!params.success || !parsed.success) { res.status(400).json({ error: "Invalid memory update" }); return; }
  const [item] = await db.update(memoriesTable).set(parsed.data).where(and(eq(memoriesTable.id, params.data.id), eq(memoriesTable.userId, currentUser(res).id))).returning();
  if (!item) { res.status(404).json({ error: "Memory not found" }); return; }
  res.json(item);
});

router.delete("/memories/:id", async (req, res): Promise<void> => {
  const params = DeleteMemoryParams.safeParse(req.params);
  if (!params.success) { res.status(400).json({ error: params.error.message }); return; }
  const [item] = await db.delete(memoriesTable).where(and(eq(memoriesTable.id, params.data.id), eq(memoriesTable.userId, currentUser(res).id))).returning();
  if (!item) { res.status(404).json({ error: "Memory not found" }); return; }
  res.sendStatus(204);
});

router.get("/documents", async (_req, res): Promise<void> => {
  const items = await db.select().from(documentsTable).where(eq(documentsTable.userId, currentUser(res).id)).orderBy(desc(documentsTable.createdAt));
  res.json(ListDocumentsResponse.parse(items.map(toDocument)));
});

router.post("/documents", async (req, res): Promise<void> => {
  const parsed = CreateDocumentBody.safeParse(req.body);
  if (!parsed.success) { res.status(400).json({ error: parsed.error.message }); return; }
  const created = await indexDocument(currentUser(res).id, parsed.data.filename, parsed.data.title, parsed.data.content, parsed.data.fileSize ?? parsed.data.content.length);
  res.status(201).json(CreateDocumentResponse.parse(toDocument(created)));
});

router.post("/documents/upload", upload.single("file"), async (req, res): Promise<void> => {
  if (!req.file) { res.status(400).json({ error: "A file is required" }); return; }
  const filename = req.file.originalname.toLowerCase();
  const hasPdfMagicBytes = req.file.buffer[0] === 0x25 && req.file.buffer[1] === 0x50 && req.file.buffer[2] === 0x44 && req.file.buffer[3] === 0x46;
  const isPdf = filename.endsWith(".pdf") || req.file.mimetype === "application/pdf" || hasPdfMagicBytes;
  const isText = req.file.mimetype.startsWith("text/") || filename.endsWith(".txt") || filename.endsWith(".md") || filename.endsWith(".markdown");
  if (!isPdf && !isText) {
    res.status(400).json({ error: "Only PDF, TXT, and MD files are supported. Check the file type and try again." }); return;
  }
  const geminiKey = process.env.GEMINI_API_KEY;
  if (!geminiKey) { res.status(500).json({ error: "Gemini API key not configured" }); return; }

  let text: string | null = null;
  let analysis: { summary: string; keyPoints: string[]; mainTopics: string[] } = { summary: "", keyPoints: [], mainTopics: [] };
  let geminiFileRef: string | null = null;

  try {
    if (isPdf) {
      // Upload to Gemini Files API
      const uploadUrl = `https://generativelanguage.googleapis.com/upload/v1beta/files?key=${geminiKey}`;
      const uploadResponse = await fetch(uploadUrl, {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: req.file.buffer,
      });
      if (!uploadResponse.ok) {
        const errText = await uploadResponse.text();
        console.error("[documents/upload] Gemini upload failed:", errText);
        res.status(422).json({ error: "PDF upload to Gemini failed." }); return;
      }
      const geminiFile = (await uploadResponse.json()) as { name: string };
      geminiFileRef = geminiFile.name;
      if (!geminiFileRef) { res.status(422).json({ error: "Gemini file reference missing." }); return; }

      // Ask Gemini to analyze the PDF
      const analysisPrompt = `Analyze this PDF document. Provide a summary, key points (array), and main topics (array) in valid JSON with keys "summary", "keyPoints", and "mainTopics". Do not invent facts not present in the text.\n\nThe PDF file reference is: ${geminiFileRef}. Analyze its content.`;
      const analysisResp = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/${process.env.GEMINI_MODEL ?? "gemini-3.6-flash"}:generateContent?key=${geminiKey}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json", "x-goog-api-key": geminiKey },
          body: JSON.stringify({
            contents: [{ role: "user", parts: [{ file: geminiFileRef }] }],
            generationConfig: { temperature: 0.35, maxOutputTokens: 8192 },
          }),
        },
      );
      if (!analysisResp.ok) {
        const errBody = await analysisResp.text();
        console.error("[documents/upload] Gemini analysis failed:", errBody);
      } else {
        const analysisBody = (await analysisResp.json()) as any;
        const candidate = analysisBody.candidates?.[0]?.content?.parts?.[0]?.text;
        if (candidate) {
          try {
            const parsed = JSON.parse(candidate);
            analysis = { summary: parsed.summary || "", keyPoints: parsed.keyPoints || [], mainTopics: parsed.mainTopics || [] };
          } catch {
            analysis = { summary: candidate, keyPoints: [], mainTopics: [] };
          }
        }
      }

      // Extract text from the PDF via Gemini
      const textPrompt = `Extract all textual content from this PDF document. Return only the raw text, no analysis. The file reference is: ${geminiFileRef}`;
      const textResp = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/${process.env.GEMINI_MODEL ?? "gemini-3.6-flash"}:generateContent?key=${geminiKey}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json", "x-goog-api-key": geminiKey },
          body: JSON.stringify({
            contents: [{ role: "user", parts: [{ text: textPrompt }] }],
            generationConfig: { temperature: 0., maxOutputTokens: 16384 },
          }),
        },
      );
      if (textResp.ok) {
        const textBody = (await textResp.json()) as any;
        const textCandidate = textBody.candidates?.[0]?.content?.parts?.[0]?.text;
        if (textCandidate) text = textCandidate;
      }
    } else {
      // TXT: existing behavior
      text = req.file.buffer.toString("utf-8");
    }
  } catch (err) {
    console.error("[documents/upload] processing error:", err);
  }

  // Ensure we have at least some text from the file
  if (!text) {
    text = isPdf ? null : req.file.buffer.toString("utf-8");
  }

  const title = typeof req.body.title === "string" && req.body.title.trim() ? req.body.title.trim() : req.file.originalname.replace(/\.(pdf|txt|md|markdown)$/i, "");

  // Store in database - use the Gemini file reference if available, otherwise stored text
  const storedText = text || "";
  const chunks = splitIntoChunks(storedText);
  const created = await indexDocument(currentUser(res).id, req.file.originalname, title, storedText.trim(), req.file.size);

  // Only mark READY if we have text and analysis
  const ready = storedText.trim().length > 0 && analysis.summary.length > 0;
  await db.update(documentsTable).set({ processingStatus: ready ? "ready" : "pending" }).where(eq(documentsTable.id, created.id));
  res.status(201).json(CreateDocumentResponse.parse(toDocument(created)));
});

router.get("/documents/:id", async (req, res): Promise<void> => {
  const params = GetDocumentParams.safeParse(req.params);
  if (!params.success) { res.status(400).json({ error: params.error.message }); return; }
  const [document] = await db.select().from(documentsTable).where(and(eq(documentsTable.id, params.data.id), eq(documentsTable.userId, currentUser(res).id))).limit(1);
  if (!document) { res.status(404).json({ error: "Document not found" }); return; }
  res.json(GetDocumentResponse.parse({ ...toDocument(document), textPreview: document.contentText.slice(0, 1000) }));
});

router.delete("/documents/:id", async (req, res): Promise<void> => {
  const params = DeleteDocumentParams.safeParse(req.params);
  if (!params.success) { res.status(400).json({ error: params.error.message }); return; }
  const [document] = await db.delete(documentsTable).where(and(eq(documentsTable.id, params.data.id), eq(documentsTable.userId, currentUser(res).id))).returning();
  if (!document) { res.status(404).json({ error: "Document not found" }); return; }
  await db.delete(documentChunksTable).where(eq(documentChunksTable.documentId, document.id));
  res.sendStatus(204);
});

router.post("/documents/query", async (req, res): Promise<void> => {
  const parsed = QueryDocumentsBody.safeParse(req.body);
  if (!parsed.success) { res.status(400).json({ error: parsed.error.message }); return; }
  const matches = await retrieveDocumentChunks(currentUser(res).id, parsed.data.query, parsed.data.documentId);
  const context = matches.map((item) => `${item.document.title}: ${item.chunk.content}`).join("\n");
  const generated = context ? await generateGemini(`Answer the question only from these document excerpts. Cite the document title in plain language. If unsupported, say so.\nQuestion: ${parsed.data.query}\nExcerpts:\n${context}`) : null;
  const answer = generated ?? (matches[0] ? `I found this in ${matches[0].document.title}: ${matches[0].chunk.content.slice(0, 500)}` : "I couldn't find a matching passage in your indexed documents.");
  res.json(QueryDocumentsResponse.parse({
    answer,
    sources: matches.map((item) => ({ documentId: item.document.id, title: item.document.title, excerpt: item.chunk.content.slice(0, 260) })),
  }));
});

router.get("/goals", async (_req, res): Promise<void> => {
  const items = await db.select().from(goalsTable).where(eq(goalsTable.userId, currentUser(res).id)).orderBy(desc(goalsTable.updatedAt));
  res.json(ListGoalsResponse.parse(items));
});

router.post("/goals", async (req, res): Promise<void> => {
  const parsed = CreateGoalBody.safeParse(req.body);
  if (!parsed.success) { res.status(400).json({ error: parsed.error.message }); return; }
  const userId = currentUser(res).id;
  const [goal] = await db.insert(goalsTable).values({ ...parsed.data, userId, deadline: parsed.data.deadline ? parsed.data.deadline.toISOString().slice(0, 10) : null }).returning();
  await Promise.all([awardXp(userId, `goal:${goal.id}`, 10), logActivity(userId, "goal", "Goal created", goal.title)]);
  res.status(201).json(CreateGoalResponse.parse(goal));
});

router.get("/goals/:id", async (req, res): Promise<void> => {
  const params = GetGoalParams.safeParse(req.params);
  if (!params.success) { res.status(400).json({ error: params.error.message }); return; }
  const [goal] = await db.select().from(goalsTable).where(and(eq(goalsTable.id, params.data.id), eq(goalsTable.userId, currentUser(res).id))).limit(1);
  if (!goal) { res.status(404).json({ error: "Goal not found" }); return; }
  const milestones = await db.select().from(milestonesTable).where(eq(milestonesTable.goalId, goal.id)).orderBy(milestonesTable.createdAt);
  res.json(GetGoalResponse.parse({ ...goal, milestones }));
});

router.patch("/goals/:id", async (req, res): Promise<void> => {
  const params = UpdateGoalParams.safeParse(req.params);
  const parsed = UpdateGoalBody.safeParse(req.body);
  if (!params.success || !parsed.success) { res.status(400).json({ error: "Invalid goal update" }); return; }
  const userId = currentUser(res).id;
  const { deadline, ...goalValues } = parsed.data;
  const values = deadline === undefined
    ? goalValues
    : { ...goalValues, deadline: deadline ? deadline.toISOString().slice(0, 10) : null };
  const [goal] = await db.update(goalsTable).set(values).where(and(eq(goalsTable.id, params.data.id), eq(goalsTable.userId, userId))).returning();
  if (!goal) { res.status(404).json({ error: "Goal not found" }); return; }
  if (goal.status === "completed") await awardXp(userId, `goal-completed:${goal.id}`, 50);
  res.json(goal);
});

router.delete("/goals/:id", async (req, res): Promise<void> => {
  const params = DeleteGoalParams.safeParse(req.params);
  if (!params.success) { res.status(400).json({ error: params.error.message }); return; }
  const [goal] = await db.delete(goalsTable).where(and(eq(goalsTable.id, params.data.id), eq(goalsTable.userId, currentUser(res).id))).returning();
  if (!goal) { res.status(404).json({ error: "Goal not found" }); return; }
  await db.delete(milestonesTable).where(eq(milestonesTable.goalId, goal.id));
  res.sendStatus(204);
});

router.post("/goals/:id/milestones", async (req, res): Promise<void> => {
  const params = req.params.id;
  const goalId = Number(params);
  const parsed = CreateMilestoneBody.safeParse(req.body);
  if (!Number.isInteger(goalId) || !parsed.success) { res.status(400).json({ error: "Invalid milestone" }); return; }
  const [goal] = await db.select().from(goalsTable).where(and(eq(goalsTable.id, goalId), eq(goalsTable.userId, currentUser(res).id))).limit(1);
  if (!goal) { res.status(404).json({ error: "Goal not found" }); return; }
  const [milestone] = await db.insert(milestonesTable).values({ ...parsed.data, goalId, dueDate: parsed.data.dueDate ? parsed.data.dueDate.toISOString().slice(0, 10) : null }).returning();
  await syncGoalProgress(goalId);
  res.status(201).json(CreateMilestoneResponse.parse(milestone));
});

router.patch("/milestones/:id", async (req, res): Promise<void> => {
  const params = UpdateMilestoneParams.safeParse(req.params);
  const parsed = UpdateMilestoneBody.safeParse(req.body);
  if (!params.success || !parsed.success) { res.status(400).json({ error: "Invalid milestone update" }); return; }
  const userId = currentUser(res).id;
  const [owned] = await db.select({ milestone: milestonesTable }).from(milestonesTable).innerJoin(goalsTable, eq(milestonesTable.goalId, goalsTable.id)).where(and(eq(milestonesTable.id, params.data.id), eq(goalsTable.userId, userId))).limit(1);
  if (!owned) { res.status(404).json({ error: "Milestone not found" }); return; }
  const { dueDate, ...milestoneValues } = parsed.data;
  const values = dueDate === undefined
    ? milestoneValues
    : { ...milestoneValues, dueDate: dueDate ? dueDate.toISOString().slice(0, 10) : null };
  const [milestone] = await db.update(milestonesTable).set(values).where(eq(milestonesTable.id, params.data.id)).returning();
  if (milestone.status === "completed") await awardXp(userId, `milestone-completed:${milestone.id}`, 25);
  await syncGoalProgress(milestone.goalId);
  res.json(milestone);
});

router.delete("/milestones/:id", async (req, res): Promise<void> => {
  const params = DeleteMilestoneParams.safeParse(req.params);
  if (!params.success) { res.status(400).json({ error: params.error.message }); return; }
  const userId = currentUser(res).id;
  const [owned] = await db.select({ milestone: milestonesTable }).from(milestonesTable).innerJoin(goalsTable, eq(milestonesTable.goalId, goalsTable.id)).where(and(eq(milestonesTable.id, params.data.id), eq(goalsTable.userId, userId))).limit(1);
  if (!owned) { res.status(404).json({ error: "Milestone not found" }); return; }
  const goalId = owned.milestone.goalId;
  await db.delete(milestonesTable).where(eq(milestonesTable.id, params.data.id));
  await syncGoalProgress(goalId);
  res.sendStatus(204);
});

router.get("/meetings", async (_req, res): Promise<void> => {
  const items = await db.select().from(meetingsTable).where(eq(meetingsTable.userId, currentUser(res).id)).orderBy(desc(meetingsTable.createdAt));
  res.json(ListMeetingsResponse.parse(items));
});

router.post("/meetings", async (req, res): Promise<void> => {
  const parsed = CreateMeetingBody.safeParse(req.body);
  if (!parsed.success) { res.status(400).json({ error: parsed.error.message }); return; }
  const userId = currentUser(res).id;
  const transcript = parsed.data.transcript.trim();
  const generated = await generateGemini(`Extract a concise meeting summary, decisions, action items, and participants from this transcript. Return JSON with keys summary, decisions, actionItems, participants. Do not invent details.\n${transcript}`);
  let extracted = { summary: transcript.slice(0, 280), decisions: [] as string[], actionItems: [] as string[], participants: [] as string[] };
  if (generated) {
    const cleaned = generated.replace(/```(?:json)?\s*/gi, "").replace(/```\s*/gi, "").trim();
    if (cleaned) {
      try {
        const json = JSON.parse(cleaned);
        extracted = {
          summary: typeof json.summary === "string" && json.summary.trim() ? json.summary : extracted.summary,
          decisions: Array.isArray(json.decisions) ? json.decisions.filter((d: unknown): d is string => typeof d === "string") : [],
          actionItems: Array.isArray(json.actionItems) ? json.actionItems.filter((a: unknown): a is string => typeof a === "string") : [],
          participants: Array.isArray(json.participants) ? json.participants.filter((p: unknown): p is string => typeof p === "string") : [],
        };
      } catch { /* keep safe fallback */ }
    }
  }
  const [meeting] = await db.insert(meetingsTable).values({
    userId, title: parsed.data.title, transcript, meetingDate: parsed.data.meetingDate ? parsed.data.meetingDate.toISOString().slice(0, 10) : null,
    summary: extracted.summary, decisions: extracted.decisions, actionItems: extracted.actionItems, participants: extracted.participants,
  }).returning();
  await Promise.all([awardXp(userId, `meeting:${meeting.id}`, 15), logActivity(userId, "meeting", "Meeting processed", meeting.title)]);
  res.status(201).json(CreateMeetingResponse.parse(meeting));
});

router.get("/meetings/:id", async (req, res): Promise<void> => {
  const params = GetMeetingParams.safeParse(req.params);
  if (!params.success) { res.status(400).json({ error: params.error.message }); return; }
  const [meeting] = await db.select().from(meetingsTable).where(and(eq(meetingsTable.id, params.data.id), eq(meetingsTable.userId, currentUser(res).id))).limit(1);
  if (!meeting) { res.status(404).json({ error: "Meeting not found" }); return; }
  res.json(GetMeetingResponse.parse(meeting));
});

router.delete("/meetings/:id", async (req, res): Promise<void> => {
  const params = DeleteMeetingParams.safeParse(req.params);
  if (!params.success) { res.status(400).json({ error: params.error.message }); return; }
  const [meeting] = await db.delete(meetingsTable).where(and(eq(meetingsTable.id, params.data.id), eq(meetingsTable.userId, currentUser(res).id))).returning();
  if (!meeting) { res.status(404).json({ error: "Meeting not found" }); return; }
  res.sendStatus(204);
});

router.get("/reminders", async (_req, res): Promise<void> => {
  const items = await db.select().from(remindersTable).where(eq(remindersTable.userId, currentUser(res).id)).orderBy(remindersTable.dueAt);
  res.json(ListRemindersResponse.parse(items));
});

router.post("/reminders", async (req, res): Promise<void> => {
  const parsed = CreateReminderBody.safeParse(req.body);
  if (!parsed.success) { res.status(400).json({ error: parsed.error.message }); return; }
  const [reminder] = await db.insert(remindersTable).values({ ...parsed.data, userId: currentUser(res).id, dueAt: parsed.data.dueAt }).returning();
  res.status(201).json(CreateReminderResponse.parse(reminder));
});

router.patch("/reminders/:id", async (req, res): Promise<void> => {
  const params = UpdateReminderParams.safeParse(req.params);
  const parsed = UpdateReminderBody.safeParse(req.body);
  if (!params.success || !parsed.success) { res.status(400).json({ error: "Invalid reminder update" }); return; }
  const [reminder] = await db.update(remindersTable).set(parsed.data).where(and(eq(remindersTable.id, params.data.id), eq(remindersTable.userId, currentUser(res).id))).returning();
  if (!reminder) { res.status(404).json({ error: "Reminder not found" }); return; }
  res.json(reminder);
});

router.delete("/reminders/:id", async (req, res): Promise<void> => {
  const params = DeleteReminderParams.safeParse(req.params);
  if (!params.success) { res.status(400).json({ error: params.error.message }); return; }
  const [reminder] = await db.delete(remindersTable).where(and(eq(remindersTable.id, params.data.id), eq(remindersTable.userId, currentUser(res).id))).returning();
  if (!reminder) { res.status(404).json({ error: "Reminder not found" }); return; }
  res.sendStatus(204);
});

router.get("/insights/dashboard", async (_req, res): Promise<void> => {
  const userId = currentUser(res).id;
  const [goals, reminders, memories, documents, meetings, xp] = await Promise.all([
    db.select().from(goalsTable).where(eq(goalsTable.userId, userId)).orderBy(desc(goalsTable.updatedAt)),
    db.select().from(remindersTable).where(and(eq(remindersTable.userId, userId), eq(remindersTable.status, "open"))).orderBy(remindersTable.dueAt).limit(4),
    db.select().from(memoriesTable).where(eq(memoriesTable.userId, userId)).orderBy(desc(memoriesTable.updatedAt)).limit(3),
    db.select().from(documentsTable).where(eq(documentsTable.userId, userId)).orderBy(desc(documentsTable.createdAt)).limit(3),
    db.select().from(meetingsTable).where(eq(meetingsTable.userId, userId)).orderBy(desc(meetingsTable.createdAt)).limit(3),
    db.select({ total: sql<number>`coalesce(sum(${xpEventsTable.amount}), 0)` }).from(xpEventsTable).where(eq(xpEventsTable.userId, userId)),
  ]);
  const active = goals.filter((goal) => goal.status === "active");
  const xpTotal = Number(xp[0]?.total ?? 0);
  const focus = active[0] ? `${active[0].title} · ${active[0].progress}% complete` : "Add a goal to give JARVIS something meaningful to prioritize.";
  res.json({
    greeting: `Good morning, ${currentUser(res).name.split(" ")[0]}`,
    focus,
    activeGoals: active.length,
    completedGoals: goals.filter((goal) => goal.status === "completed").length,
    upcomingReminders: reminders,
    recentMemories: memories,
    recentDocuments: documents.map(toDocument),
    recentMeetings: meetings,
    xp: xpTotal,
    level: levelForXp(xpTotal),
    streak: await calculateStreak(userId),
  });
});

router.get("/insights/growth", async (_req, res): Promise<void> => {
  const userId = currentUser(res).id;
  const [goals, memories, documents, meetings, xp, activities] = await Promise.all([
    db.select().from(goalsTable).where(eq(goalsTable.userId, userId)),
    db.select().from(memoriesTable).where(eq(memoriesTable.userId, userId)),
    db.select().from(documentsTable).where(eq(documentsTable.userId, userId)),
    db.select().from(meetingsTable).where(eq(meetingsTable.userId, userId)),
    db.select({ total: sql<number>`coalesce(sum(${xpEventsTable.amount}), 0)` }).from(xpEventsTable).where(eq(xpEventsTable.userId, userId)),
    db.select().from(activityEventsTable).where(eq(activityEventsTable.userId, userId)),
  ]);
  const xpTotal = Number(xp[0]?.total ?? 0);
  const completedGoals = goals.filter((goal) => goal.status === "completed").length;
  const skills = [...new Set([
    ...goals.map((goal) => goal.category),
    ...memories.filter((item) => item.category === "interest" || item.category === "education").map((item) => item.content.split(" ").slice(0, 2).join(" ")),
  ])].filter(Boolean).slice(0, 8);
  res.json({
    xp: xpTotal, level: levelForXp(xpTotal), nextLevelXp: nextLevelXp(levelForXp(xpTotal)), streak: await calculateStreak(userId),
    activeGoals: goals.filter((goal) => goal.status === "active").length, completedGoals,
    completionRate: goals.length ? Math.round((completedGoals / goals.length) * 100) : 0,
    completedTasks: 0, memoriesCreated: memories.length, documentsUploaded: documents.length,
    meetingsProcessed: meetings.length, skills: skills.length ? skills : ["Build your first goal to reveal skills"],
  });
});

router.get("/insights/timeline", async (_req, res): Promise<void> => {
  const items = await db.select().from(activityEventsTable).where(eq(activityEventsTable.userId, currentUser(res).id)).orderBy(desc(activityEventsTable.occurredAt)).limit(80);
  res.json(items.map((item) => ({ id: String(item.id), kind: item.kind, title: item.title, description: item.description, occurredAt: item.occurredAt })));
});

router.get("/insights/knowledge", async (req, res): Promise<void> => {
  const parsed = SearchKnowledgeQueryParams.safeParse(req.query);
  if (!parsed.success) { res.status(400).json({ error: parsed.error.message }); return; }
  res.json(SearchKnowledgeResponse.parse(await searchKnowledge(currentUser(res).id, parsed.data.q)));
});

async function indexDocument(userId: number, filename: string, title: string, content: string, fileSize: number) {
  const chunks = splitIntoChunks(content);
  const [document] = await db.insert(documentsTable).values({
    userId, filename, title, fileSize, contentText: content, chunkCount: chunks.length, processingStatus: "pending",
  }).returning();
  if (chunks.length) await db.insert(documentChunksTable).values(chunks.map((chunk, index) => ({ documentId: document.id, chunkIndex: index, content: chunk })));
  await Promise.all([awardXp(userId, `document:${document.id}`, 10), logActivity(userId, "document", "Document indexed", title)]);
  return document;
}

async function extractPdfText(buffer: Buffer): Promise<string> {
  const { getDocument } = await import("pdfjs-dist/legacy/build/pdf.mjs");
  const { createRequire } = await import("node:module");
  const require = createRequire(import.meta.url);
  const pdfjsDir = require("node:path").dirname(require.resolve("pdfjs-dist/package.json"));
  const pdf = await getDocument({
    data: new Uint8Array(buffer),
    disableWorker: true,
    standardFontDataUrl: require("node:path").join(pdfjsDir, "standard_fonts") + "/",
    cMapUrl: require("node:path").join(pdfjsDir, "cmaps") + "/",
  } as any).promise;
  const pages: string[] = [];
  for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber += 1) {
    const page = await pdf.getPage(pageNumber);
    const content = await page.getTextContent();
    pages.push(content.items.map((item) => ("str" in item ? item.str : "")).join(" "));
  }
  return pages.join("\n").replace(/\s+/g, " ").trim();
}

function toDocument(document: typeof documentsTable.$inferSelect) {
  return {
    id: document.id,
    filename: document.filename,
    title: document.title,
    fileSize: document.fileSize,
    processingStatus: document.processingStatus,
    chunkCount: document.chunkCount,
    createdAt: document.createdAt,
  };
}

async function calculateStreak(userId: number): Promise<number> {
  const items = await db.select({ occurredAt: activityEventsTable.occurredAt }).from(activityEventsTable).where(eq(activityEventsTable.userId, userId)).orderBy(desc(activityEventsTable.occurredAt)).limit(30);
  const days = new Set(items.map((item) => item.occurredAt.toISOString().slice(0, 10)));
  let streak = 0;
  const cursor = new Date();
  while (days.has(cursor.toISOString().slice(0, 10))) {
    streak += 1;
    cursor.setUTCDate(cursor.getUTCDate() - 1);
  }
  return streak;
}

export default router;
