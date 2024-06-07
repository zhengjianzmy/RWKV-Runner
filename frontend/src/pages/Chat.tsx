import React, { FC, useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Avatar, Button, Menu, MenuPopover, MenuTrigger, PresenceBadge, Textarea } from '@fluentui/react-components';
import commonStore, { ModelStatus } from '../stores/commonStore';
import { observer } from 'mobx-react-lite';
import { v4 as uuid } from 'uuid';
import classnames from 'classnames';
import { fetchEventSource } from '@microsoft/fetch-event-source';
import { KebabHorizontalIcon, PencilIcon, SyncIcon, TrashIcon } from '@primer/octicons-react';
import logo from '../assets/images/logo.png';
import MarkdownRender from '../components/MarkdownRender';
import { ToolTipButton } from '../components/ToolTipButton';
import {
  ArrowCircleUp28Regular,
  ArrowClockwise16Regular,
  Attach16Regular,
  Delete28Regular,
  Dismiss16Regular,
  Dismiss24Regular,
  RecordStop28Regular,
  SaveRegular,
  TextAlignJustify24Regular,
  TextAlignJustifyRotate9024Regular
} from '@fluentui/react-icons';
import { CopyButton } from '../components/CopyButton';
import { ReadButton } from '../components/ReadButton';
import { toast } from 'react-toastify';
import { WorkHeader } from '../components/WorkHeader';
import { DialogButton } from '../components/DialogButton';
import { OpenFileFolder, OpenOpenFileDialog, OpenSaveFileDialog } from '../../wailsjs/go/backend_golang/App';
import { absPathAsset, bytesToReadable, getServerRoot, setActivePreset, toastWithButton } from '../utils';
import { useMediaQuery } from 'usehooks-ts';
import { botName, ConversationMessage, MessageType, userName, welcomeUuid } from '../types/chat';
import { Labeled } from '../components/Labeled';
import { ValuedSlider } from '../components/ValuedSlider';
import { PresetsButton } from './PresetsManager/PresetsButton';
import { webOpenOpenFileDialog } from '../utils/web-file-operations';
import { Buffer } from 'buffer';

let chatSseControllers: {
  [id: string]: AbortController
} = {};

const MoreUtilsButton: FC<{
  uuid: string,
  setEditing: (editing: boolean) => void
}> = observer(({
  uuid,
  setEditing
}) => {
  const { t } = useTranslation();
  const [speaking, setSpeaking] = useState(false);

  const messageItem = commonStore.conversation[uuid];

  return <Menu>
    <MenuTrigger disableButtonEnhancement>
      <Button icon={<KebabHorizontalIcon />} size="small" appearance="subtle" />
    </MenuTrigger>
    <MenuPopover style={{ minWidth: 0 }}>
      <CopyButton content={messageItem.content} showDelay={500} />
      <ReadButton content={messageItem.content} inSpeaking={speaking} showDelay={500} setSpeakingOuter={setSpeaking} />
      <ToolTipButton desc={t('Edit')} icon={<PencilIcon />} showDelay={500} size="small" appearance="subtle"
        onClick={() => {
          setEditing(true);
        }} />
      <ToolTipButton desc={t('Delete')} icon={<TrashIcon />} showDelay={500} size="small" appearance="subtle"
        onClick={() => {
          commonStore.conversationOrder.splice(commonStore.conversationOrder.indexOf(uuid), 1);
          delete commonStore.conversation[uuid];
          commonStore.setAttachment(uuid, null);
        }} />
    </MenuPopover>
  </Menu>;
});

const ChatMessageItem: FC<{
  uuid: string,
  onSubmit: (message: string | null, answerId: string | null,
    startUuid: string | null, endUuid: string | null, includeEndUuid: boolean) => void
}> = observer(({ uuid, onSubmit }) => {
  const { t } = useTranslation();
  const [editing, setEditing] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const messageItem = commonStore.conversation[uuid];

  // console.log(uuid);

  const setEditingInner = (editing: boolean) => {
    setEditing(editing);
    if (editing) {
      setTimeout(() => {
        const textarea = textareaRef.current;
        if (textarea) {
          textarea.focus();
          textarea.selectionStart = textarea.value.length;
          textarea.selectionEnd = textarea.value.length;
          textarea.style.height = textarea.scrollHeight + 'px';
        }
      });
    }
  };

  let avatarImg: string | undefined;
  if (commonStore.activePreset && messageItem.sender === botName) {
    avatarImg = absPathAsset(commonStore.activePreset.avatarImg);
  } else if (messageItem.avatarImg) {
    avatarImg = messageItem.avatarImg;
  }

  return <div
    className={classnames(
      'flex gap-2 mb-2 overflow-hidden',
      messageItem.side === 'left' ? 'flex-row' : 'flex-row-reverse'
    )}
    onMouseEnter={() => {
      const utils = document.getElementById('utils-' + uuid);
      if (utils) utils.classList.remove('invisible');
    }}
    onMouseLeave={() => {
      const utils = document.getElementById('utils-' + uuid);
      if (utils) utils.classList.add('invisible');
    }}
  >
    <Avatar
      color={messageItem.color}
      name={messageItem.sender}
      image={avatarImg ? { src: avatarImg } : undefined}
    />
    <div
      className={classnames(
        'flex p-2 rounded-lg overflow-hidden',
        editing ? 'grow' : '',
        messageItem.side === 'left' ? 'bg-gray-200' : 'bg-blue-500',
        messageItem.side === 'left' ? 'text-gray-600' : 'text-white'
      )}
    >
      {!editing ?
        <div className="flex flex-col">
          <MarkdownRender>{messageItem.content}</MarkdownRender>
          {uuid in commonStore.attachments &&
            <div className="flex grow">
              <div className="grow" />
              <ToolTipButton className="whitespace-nowrap"
                text={
                  commonStore.attachments[uuid][0].name.replace(
                    new RegExp('(^[^\\.]{5})[^\\.]+'), '$1...')
                }
                desc={`${commonStore.attachments[uuid][0].name} (${bytesToReadable(commonStore.attachments[uuid][0].size)})`}
                size="small" shape="circular" appearance="secondary" />
            </div>
          }
        </div> :
        <Textarea ref={textareaRef}
          className="grow"
          style={{ minWidth: 0 }}
          value={messageItem.content}
          onChange={(e) => {
            messageItem.content = e.target.value;
            commonStore.conversation[uuid].type = MessageType.Normal;
            commonStore.conversation[uuid].done = true;
            commonStore.setConversation(commonStore.conversation);
            commonStore.setConversationOrder([...commonStore.conversationOrder]);
          }}
          onBlur={() => {
            setEditingInner(false);
          }} />}
    </div>
    <div className="flex flex-col gap-1 items-start">
      <div className="grow" />
      {(messageItem.type === MessageType.Error || !messageItem.done) &&
        <PresenceBadge size="extra-small" status={
          messageItem.type === MessageType.Error ? 'busy' : 'away'
        } />
      }
      <div className="flex invisible" id={'utils-' + uuid}>
        {
          messageItem.sender === botName && uuid !== welcomeUuid &&
          <ToolTipButton desc={t('Retry')} size="small" appearance="subtle"
            icon={<SyncIcon />} onClick={() => {
            if (uuid in chatSseControllers) {
              chatSseControllers[uuid].abort();
              delete chatSseControllers[uuid];
            }
            onSubmit(null, uuid, null, uuid, false);
          }} />
        }
        <ToolTipButton desc={t('Edit')} icon={<PencilIcon />} size="small" appearance="subtle"
          onClick={() => {
            setEditingInner(true);
          }} />
        <MoreUtilsButton uuid={uuid} setEditing={setEditingInner} />
      </div>
    </div>
  </div>;
});

const SidePanel: FC = observer(() => {
  const [t] = useTranslation();
  const mq = useMediaQuery('(min-width: 640px)');
  const params = commonStore.chatParams;

  return <div
    className={classnames(
      'flex flex-col gap-1 h-full flex-shrink-0 transition-width duration-300 ease-in-out',
      commonStore.sidePanelCollapsed ? 'w-0' : (mq ? 'w-64' : 'w-full'),
      !commonStore.sidePanelCollapsed && 'ml-1')
    }>
    <div className="flex m-1">
      <div className="grow" />
      <PresetsButton tab="Chat" size="medium" shape="circular" appearance="subtle" />
      <Button size="medium" shape="circular" appearance="subtle" icon={<Dismiss24Regular />}
        onClick={() => commonStore.setSidePanelCollapsed(true)}
      />
    </div>
    <div className="flex flex-col gap-1 overflow-x-hidden overflow-y-auto p-1">
      <Labeled flex breakline label={t('Max Response Token')}
        desc={t('By default, the maximum number of tokens that can be answered in a single response, it can be changed by the user by specifying API parameters.')}
        content={
          <ValuedSlider value={params.maxResponseToken} min={100} max={8100}
            step={100}
            input
            onChange={(e, data) => {
              commonStore.setChatParams({
                maxResponseToken: data.value
              });
            }} />
        } />
      <Labeled flex breakline label={t('Temperature')}
        desc={t('Sampling temperature, it\'s like giving alcohol to a model, the higher the stronger the randomness and creativity, while the lower, the more focused and deterministic it will be.')}
        content={
          <ValuedSlider value={params.temperature} min={0} max={2} step={0.1}
            input
            onChange={(e, data) => {
              commonStore.setChatParams({
                temperature: data.value
              });
            }} />
        } />
      <Labeled flex breakline label={t('Top_P')}
        desc={t('Just like feeding sedatives to the model. Consider the results of the top n% probability mass, 0.1 considers the top 10%, with higher quality but more conservative, 1 considers all results, with lower quality but more diverse.')}
        content={
          <ValuedSlider value={params.topP} min={0} max={1} step={0.1} input
            onChange={(e, data) => {
              commonStore.setChatParams({
                topP: data.value
              });
            }} />
        } />
      <Labeled flex breakline label={t('Presence Penalty')}
        desc={t('Positive values penalize new tokens based on whether they appear in the text so far, increasing the model\'s likelihood to talk about new topics.')}
        content={
          <ValuedSlider value={params.presencePenalty} min={0} max={2}
            step={0.1} input
            onChange={(e, data) => {
              commonStore.setChatParams({
                presencePenalty: data.value
              });
            }} />
        } />
      <Labeled flex breakline label={t('Frequency Penalty')}
        desc={t('Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model\'s likelihood to repeat the same line verbatim.')}
        content={
          <ValuedSlider value={params.frequencyPenalty} min={0} max={2}
            step={0.1} input
            onChange={(e, data) => {
              commonStore.setChatParams({
                frequencyPenalty: data.value
              });
            }} />
        } />
    </div>
    <div className="grow" />
    {/*<Button*/}
    {/*  icon={<FolderOpenVerticalRegular />}*/}
    {/*  onClick={() => {*/}
    {/*  }}>*/}
    {/*  {t('Load Conversation')}*/}
    {/*</Button>*/}
    <Button
      icon={<SaveRegular />}
      onClick={() => {
        let savedContent: string = '';
        const isWorldModel = commonStore.getCurrentModelConfig().modelParameters.modelName.toLowerCase().includes('world');
        const user = isWorldModel ? 'User' : 'Bob';
        const bot = isWorldModel ? 'Assistant' : 'Alice';
        commonStore.conversationOrder.forEach((uuid) => {
          if (uuid === welcomeUuid)
            return;
          const messageItem = commonStore.conversation[uuid];
          if (messageItem.type !== MessageType.Error) {
            savedContent += `${messageItem.sender === userName ? user : bot}: ${messageItem.content}\n\n`;
          }
        });

        OpenSaveFileDialog('*.txt', 'conversation.txt', savedContent).then((path) => {
          if (path)
            toastWithButton(t('Conversation Saved'), t('Open'), () => {
              OpenFileFolder(path, false);
            });
        }).catch(e => {
          toast(t('Error') + ' - ' + (e.message || e), { type: 'error', autoClose: 2500 });
        });
      }}>
      {t('Save Conversation')}
    </Button>
  </div>;
});

const ChatPanel: FC = observer(() => {
  const { t } = useTranslation();
  const bodyRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const mq = useMediaQuery('(min-width: 640px)');
  if (commonStore.sidePanelCollapsed === 'auto')
    commonStore.setSidePanelCollapsed(!mq);
  const currentConfig = commonStore.getCurrentModelConfig();
  const apiParams = currentConfig.apiParameters;
  const port = apiParams.apiPort;
  const generating: boolean = Object.keys(chatSseControllers).length > 0;

  useEffect(() => {
    if (inputRef.current)
      inputRef.current.style.maxHeight = '16rem';
    scrollToBottom();
  }, []);

  useEffect(() => {
    if (commonStore.conversationOrder.length === 0) {
      commonStore.setConversationOrder([welcomeUuid]);
      commonStore.setConversation({
        [welcomeUuid]: {
          sender: botName,
          type: MessageType.Normal,
          color: 'colorful',
          avatarImg: logo,
          time: new Date().toISOString(),
          content: commonStore.platform === 'web' ? t('Hello, what can I do for you?') : t('Hello! I\'m RWKV, an open-source and commercially usable large language model.'),
          side: 'left',
          done: true
        }
      });
    }
  }, []);

  const scrollToBottom = () => {
    if (bodyRef.current)
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
  };

  const handleKeyDownOrClick = (e: any) => {
    e.stopPropagation();
    if (e.type === 'click' || (e.keyCode === 13 && !e.shiftKey)) {
      e.preventDefault();
      if (commonStore.status.status === ModelStatus.Offline && !commonStore.settings.apiUrl && commonStore.platform !== 'web') {
        toast(t('Please click the button in the top right corner to start the model'), { type: 'warning' });
        return;
      }
      if (!commonStore.currentInput) return;
      onSubmit(commonStore.currentInput);
      commonStore.setCurrentInput('');
    }
  };

  // if message is not null, create a user message;
  // if answerId is not null, override the answer with new response;
  // if startUuid is null, start generating api body messages from first message;
  // if endUuid is null, generate api body messages until last message;
  const [shouldExecuteSecond, setShouldExecuteSecond] = useState(true);
  let tencentresponse = '';
  const onSubmit = useCallback((message: string | null = null, answerId: string | null = null,
    startUuid: string | null = null, endUuid: string | null = null, includeEndUuid: boolean = false) => {

    if (commonStore.settings.username == '' || commonStore.settings.phoneNumber == '') {
      console.log(commonStore.settings.username)
      console.log("请先登录")
      window.alert(t('Login First'));
      return;
    }

    if (message) {
      const newId = uuid();
      commonStore.conversation[newId] = {
        sender: userName,
        type: MessageType.Normal,
        color: 'brand',
        time: new Date().toISOString(),
        content: message,
        side: 'right',
        done: true
      };
      commonStore.setConversation(commonStore.conversation);
      commonStore.conversationOrder.push(newId);
      commonStore.setConversationOrder(commonStore.conversationOrder);

      if (commonStore.currentTempAttachment) {
        commonStore.setAttachment(newId, [commonStore.currentTempAttachment]);
        commonStore.setCurrentTempAttachment(null);
      }
    }

    let startIndex = startUuid ? commonStore.conversationOrder.indexOf(startUuid) : 0;
    let endIndex = endUuid ? (commonStore.conversationOrder.indexOf(endUuid) + (includeEndUuid ? 1 : 0)) : commonStore.conversationOrder.length;
    let targetRange = commonStore.conversationOrder.slice(startIndex, endIndex);

    const messages: ConversationMessage[] = [];
    targetRange.forEach((uuid, index) => {
      if (uuid === welcomeUuid)
        return;
      const messageItem = commonStore.conversation[uuid];
      if (uuid in commonStore.attachments) {
        const attachment = commonStore.attachments[uuid][0];
        messages.push({
          role: 'user',
          content: t('The content of file') + ` "${attachment.name}" `
            + t('is as follows. When replying to me, consider the file content and respond accordingly:')
            + '\n\n' + attachment.content
        });
        messages.push({ role: 'user', content: t('What\'s the file name') });
        messages.push({ role: 'assistant', content: t('The file name is: ') + attachment.name });
      }
      if (messageItem.done && messageItem.type === MessageType.Normal && messageItem.sender === userName) {
        messages.push({ role: 'user', content: messageItem.content });
      } else if (messageItem.done && messageItem.type === MessageType.Normal && messageItem.sender === botName) {
        messages.push({ role: 'assistant', content: messageItem.content });
      }
    });

    if (answerId === null) {
      answerId = uuid();
      commonStore.conversationOrder.push(answerId);
    }
    commonStore.conversation[answerId] = {
      sender: botName,
      type: MessageType.Normal,
      color: 'colorful',
      avatarImg: logo,
      time: new Date().toISOString(),
      content: '',
      side: 'left',
      done: false
    };
    commonStore.setConversation(commonStore.conversation);
    commonStore.setConversationOrder(commonStore.conversationOrder);
    setTimeout(scrollToBottom);
    let answer = '';
    const chatSseController = new AbortController();
    chatSseControllers[answerId] = chatSseController;
    let tencentcloudresult = '';
    let tencentcloudoutput = '';
    fetchEventSource( // https://api.openai.com/v1/chat/completions || http://127.0.0.1:${port}/v1/chat/completions
      getServerRoot(port, true) + '/v1/chat/completions',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${commonStore.settings.apiKey}`
        },
        body: JSON.stringify({
          messages,
          stream: true,
          model: commonStore.settings.apiChatModelName, // 'gpt-3.5-turbo'
          temperature: commonStore.chatParams.temperature,
          top_p: commonStore.chatParams.topP,
          presence_penalty: commonStore.chatParams.presencePenalty,
          frequency_penalty: commonStore.chatParams.frequencyPenalty,
          user_name: commonStore.activePreset?.userName || undefined,
          assistant_name: commonStore.activePreset?.assistantName || undefined,
          presystem: commonStore.activePreset?.presystem && undefined
        }),
        signal: chatSseController?.signal,
        onmessage(e) {
          console.log(e)
          scrollToBottom();
          if (e.data.trim() === '[DONE]') {
            if (answerId! in chatSseControllers)
              delete chatSseControllers[answerId!];
            commonStore.conversation[answerId!].done = true;
            commonStore.conversation[answerId!].content = commonStore.conversation[answerId!].content.trim();
            commonStore.setConversation(commonStore.conversation);
            commonStore.setConversationOrder([...commonStore.conversationOrder]);
            return;
          }
          let data;
          try {
            data = JSON.parse(e.data);
          } catch (error) {
            console.debug('json error', error);
            return;
          }
          if (data.model)
            commonStore.setLastModelName(data.model);
          if(data.tencentcloudresult) {
            tencentcloudresult = data.tencentcloudresult;
            
            
            // console.log(tencentcloudresult);
            // window.alert(data.tencentcloudresult);
          }
          if(data.tencentcloudoutput) {
            tencentcloudoutput = data.tencentcloudoutput;
            console.log(tencentcloudresult);
          }
          if(data.response) {
            tencentresponse = data.response;
            // console.log(data.response);
          }
          if (data.choices && Array.isArray(data.choices) && data.choices.length > 0) {
            answer += data.choices[0]?.delta?.content || '';
            commonStore.conversation[answerId!].content = answer;
            commonStore.setConversation(commonStore.conversation);
            commonStore.setConversationOrder([...commonStore.conversationOrder]);
          }
        },
        async onopen(response) {
          if (response.status !== 200) {
            commonStore.conversation[answerId!].content += '\n[ERROR]\n```\n' + response.statusText + '\n' + (await response.text()) + '\n```';
            commonStore.setConversation(commonStore.conversation);
            commonStore.setConversationOrder([...commonStore.conversationOrder]);
            setTimeout(scrollToBottom);
          }
        },
        onclose() {
          if (answerId! in chatSseControllers)
            delete chatSseControllers[answerId!];
          // console.log(tencentcloudresult);
          // window.alert(tencentcloudresult);
          // console.log(tencentcloudoutput);
          // fetchEventSource(
          //   getServerRoot(port, true) + '/v1/insert_chat',
          //   {
          //     method: 'POST',
          //     headers: {
          //       'Content-Type': 'application/json',
          //       Authorization: `Bearer ${commonStore.settings.apiKey}`
          //     },
          //     body: JSON.stringify({
          //       id: commonStore.settings.uuid,
          //       input: messages[messages.length - 1].content.replace(/"/g, "").replace(/'/g, ""),
          //       filtered_input: JSON.stringify(tencentcloudresult),
          //       output: tencentresponse.replace(/"/g, "").replace(/'/g, ""),
          //       filtered_output: JSON.stringify(tencentcloudoutput)
          //     }),
          //     onmessage(e) {
          //       // console.log(e)
          //       let data;
          //       try {
          //         data = JSON.parse(e.data);
          //       } catch (error) {
          //         console.debug('json error', error);
          //         return;
          //       }
          //     },
          //     async onopen(response) {
          //       console.log(response)
          //     },
          //     onclose() {
          //       console.log('Connection closed');
          //     },
          //     onerror(err) {
          //       throw err;
          //     }
          //   });
          console.log('Connection closed');
        },
        onerror(err) {
          if (answerId! in chatSseControllers)
            delete chatSseControllers[answerId!];
          commonStore.conversation[answerId!].type = MessageType.Error;
          commonStore.conversation[answerId!].done = true;
          err = err.message || err;
          if (err && !err.includes('ReadableStreamDefaultReader'))
            commonStore.conversation[answerId!].content += '\n[ERROR]\n```\n' + err + '\n```';
          commonStore.setConversation(commonStore.conversation);
          commonStore.setConversationOrder([...commonStore.conversationOrder]);
          setTimeout(scrollToBottom);
          throw err;
        }
      });
  }, [shouldExecuteSecond]);

  return (
    <div className="flex h-full grow pt-4 overflow-hidden">
      <div className="relative flex flex-col w-full grow gap-4 overflow-hidden">
        <Button className="absolute top-1 right-1" size="medium" shape="circular" appearance="subtle"
          style={{ zIndex: 1 }}
          icon={commonStore.sidePanelCollapsed ? <TextAlignJustify24Regular /> : <TextAlignJustifyRotate9024Regular />}
          onClick={() => commonStore.setSidePanelCollapsed(!commonStore.sidePanelCollapsed)} />
        <div ref={bodyRef} className="grow overflow-y-scroll overflow-x-hidden pr-2">
          {commonStore.conversationOrder.map(uuid =>
            <ChatMessageItem key={uuid} uuid={uuid} onSubmit={onSubmit} />
          )}
        </div>
        <div className={classnames('flex items-end', mq ? 'gap-2' : '')}>
          <DialogButton tooltip={t('Clear')}
            icon={<Delete28Regular />}
            size={mq ? 'large' : 'small'} shape="circular" appearance="subtle" title={t('Clear')}
            contentText={t('Are you sure you want to clear the conversation? It cannot be undone.')}
            onConfirm={() => {
              if (generating) {
                for (const id in chatSseControllers) {
                  chatSseControllers[id].abort();
                }
                chatSseControllers = {};
              }
              setActivePreset(commonStore.activePreset);
            }} />
          <div className="relative flex grow">
            <Textarea
              ref={inputRef}
              style={{ minWidth: 0 }}
              className="grow"
              resize="vertical"
              placeholder={t('Type your message here')!}
              value={commonStore.currentInput}
              onChange={(e) => commonStore.setCurrentInput(e.target.value)}
              onKeyDown={handleKeyDownOrClick}
            />
            <div className="absolute right-2 bottom-2">
              {!commonStore.currentTempAttachment ?
                <ToolTipButton
                  desc={commonStore.attachmentUploading ?
                    t('Processing Attachment') :
                    t('Add An Attachment (Accepts pdf, txt)')}
                  icon={commonStore.attachmentUploading ?
                    <ArrowClockwise16Regular className="animate-spin" />
                    : <Attach16Regular />}
                  size="small" shape="circular" appearance="secondary"
                  onClick={() => {
                    if (commonStore.attachmentUploading)
                      return;

                    const filterPattern = '*.txt;*.pdf';
                    const setUploading = () => commonStore.setAttachmentUploading(true);
                    // actually, status of web platform is always Offline
                    if (commonStore.platform === 'web' || commonStore.status.status === ModelStatus.Offline || currentConfig.modelParameters.device === 'WebGPU') {
                      webOpenOpenFileDialog(filterPattern, setUploading).then(webReturn => {
                        if (webReturn.content)
                          commonStore.setCurrentTempAttachment(
                            {
                              name: webReturn.blob.name,
                              size: webReturn.blob.size,
                              content: webReturn.content
                            });
                        else
                          toast(t('File is empty'), {
                            type: 'info',
                            autoClose: 1000
                          });
                      }).catch(e => {
                        toast(t('Error') + ' - ' + (e.message || e), { type: 'error', autoClose: 2500 });
                      }).finally(() => {
                        commonStore.setAttachmentUploading(false);
                      });
                    } else {
                      OpenOpenFileDialog(filterPattern).then(async filePath => {
                        if (!filePath)
                          return;

                        setUploading();

                        // Both are slow. Communication between frontend and backend is slow. Use AssetServer Handler to read the file.
                        // const blob = new Blob([atob(info.content as unknown as string)]); // await fetch(`data:application/octet-stream;base64,${info.content}`).then(r => r.blob());
                        const blob = await fetch(absPathAsset(filePath)).then(r => r.blob());
                        const attachmentName = filePath.split(/[\\/]/).pop();
                        const urlPath = `/file-to-text?file_name=${attachmentName}`;
                        const bodyForm = new FormData();
                        bodyForm.append('file_data', blob, attachmentName);
                        fetch(getServerRoot(port) + urlPath, {
                          method: 'POST',
                          body: bodyForm
                        }).then(async r => {
                            if (r.status === 200) {
                              const pages = (await r.json()).pages as any[];
                              let attachmentContent: string;
                              if (pages.length === 1)
                                attachmentContent = pages[0].page_content;
                              else
                                attachmentContent = pages.map((p, i) => `Page ${i + 1}:\n${p.page_content}`).join('\n\n');
                              if (attachmentContent)
                                commonStore.setCurrentTempAttachment(
                                  {
                                    name: attachmentName!,
                                    size: blob.size,
                                    content: attachmentContent!
                                  });
                              else
                                toast(t('File is empty'), {
                                  type: 'info',
                                  autoClose: 1000
                                });
                            } else {
                              toast(r.statusText + '\n' + (await r.text()), {
                                type: 'error'
                              });
                            }
                          }
                        ).catch(e => {
                          toast(t('Error') + ' - ' + (e.message || e), { type: 'error', autoClose: 2500 });
                        }).finally(() => {
                          commonStore.setAttachmentUploading(false);
                        });
                      }).catch(e => {
                        toast(t('Error') + ' - ' + (e.message || e), { type: 'error', autoClose: 2500 });
                      });
                    }
                  }}
                /> :
                <div>
                  <ToolTipButton
                    text={
                      commonStore.currentTempAttachment.name.replace(
                        new RegExp('(^[^\\.]{5})[^\\.]+'), '$1...')
                    }
                    desc={`${commonStore.currentTempAttachment.name} (${bytesToReadable(commonStore.currentTempAttachment.size)})`}
                    size="small" shape="circular" appearance="secondary" />
                  <ToolTipButton desc={t('Remove Attachment')}
                    icon={<Dismiss16Regular />}
                    size="small" shape="circular" appearance="subtle"
                    onClick={() => {
                      commonStore.setCurrentTempAttachment(null);
                    }} />
                </div>
              }
            </div>
          </div>
          <ToolTipButton desc={generating ? t('Stop') : t('Send')}
            icon={generating ? <RecordStop28Regular /> : <ArrowCircleUp28Regular />}
            size={mq ? 'large' : 'small'} shape="circular" appearance="subtle"
            onClick={(e) => {
              if (generating) {
                for (const id in chatSseControllers) {
                  chatSseControllers[id].abort();
                  commonStore.conversation[id].type = MessageType.Error;
                  commonStore.conversation[id].done = true;
                }
                chatSseControllers = {};
                commonStore.setConversation(commonStore.conversation);
                commonStore.setConversationOrder([...commonStore.conversationOrder]);
              } else {
                handleKeyDownOrClick(e);
              }
            }} />
        </div>
      </div>
      {/* <SidePanel /> */}
    </div>
  );
});

const Chat: FC = observer(() => {
  return (
    <div className="flex flex-col gap-1 p-2 h-full overflow-hidden">
      <WorkHeader />
      <ChatPanel />
    </div>
  );
});

export default Chat;
