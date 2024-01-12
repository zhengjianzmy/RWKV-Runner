import { FC, lazy, LazyExoticComponent, ReactElement } from 'react';
import {
  ArrowDownload20Regular,
  Chat20Regular,
  ClipboardEdit20Regular,
  DataUsageSettings20Regular,
  DocumentSettings20Regular,
  Home20Regular,
  Info20Regular,
  MusicNote220Regular,
  Settings20Regular,
  Storage20Regular
} from '@fluentui/react-icons';

type NavigationItem = {
  label: string;
  path: string;
  icon: ReactElement;
  element: LazyExoticComponent<FC>;
  top: boolean;
};

export const pages: NavigationItem[] = [
  {
    label: 'Chat',
    path: '/chat',
    icon: <Chat20Regular />,
    element: lazy(() => import('./Chat')),
    top: true
  }
];
